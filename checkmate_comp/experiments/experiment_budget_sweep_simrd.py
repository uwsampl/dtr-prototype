import argparse
import logging
import os
import pathlib
import pickle
import shutil
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import seaborn as sns
import tensorflow as tf
from matplotlib.lines import Line2D
from scipy.stats.mstats import gmean

from experiments.common.definitions import remat_data_dir
from experiments.common.graph_plotting import render_dfgraph
from experiments.common.load_keras_model import MODEL_NAMES, get_keras_model, CHAIN_GRAPH_MODELS
from experiments.common.profile.cost_model import CostModel
from experiments.common.profile.platforms import PLATFORM_CHOICES, platform_memory, pretty_platform_name
from experiments.common.ray_utils import get_futures
from remat.core.dfgraph import DFGraph
from remat.core.enum_strategy import SolveStrategy
from remat.core.schedule import ScheduledResult
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from remat.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from remat.core.solvers.strategy_chen import solve_chen_sqrtn, solve_chen_greedy
from remat.core.solvers.strategy_griewank import solve_griewank, clean_griewank_cache
from remat.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
from remat.tensorflow2.extraction import dfgraph_from_keras

from remat.core.solvers.strategy_simrd import solve_simrd
from simrd.heuristic import DTR, DTREqClass, DTRLocal, MSPS, LRU, LargestStorage, RandomStorage
from simrd.runtime import RuntimeV2EagerOptimized

NUM_ILP_CORES = os.environ.get("ILP_CORES", 12 if os.cpu_count() > 12 else 4)

SIMRD_LIVENESS = True
SIMRD_HEURISTICS = [
    DTR(), DTREqClass(), DTRLocal(), MSPS(), LRU(), LargestStorage(), RandomStorage()
]

def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', default="flops", choices=PLATFORM_CHOICES)
    parser.add_argument('--model-name', default="VGG16", choices=list(sorted(MODEL_NAMES)))
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])

    _args = parser.parse_args()
    _args.input_shape = _args.input_shape if _args.input_shape else None
    return _args


def prefix_min_np(values: np.ndarray):
    assert values.ndim == 1
    values_min = np.copy(values)
    for i in range(1, values.shape[0]):
        values_min[i] = min(values_min[i - 1], values[i])
    return values_min


def run_simrd(g, heuristic, budgets, liveness):
    logger.info('Evaluating simrd ({}), liveness {}...'.format(
        type(heuristic).__name__, 'enabled' if liveness else 'disabled'
    ))
    futures = []
    remote_simrd = ray.remote(num_cpus=NUM_ILP_CORES)(solve_simrd).remote
    for b in budgets:
        future = remote_simrd(
            g, b, heuristic=heuristic, runtime=RuntimeV2EagerOptimized,
            thrash=2.0, liveness=liveness
        )
        futures.append(future)
    results = get_futures(futures, desc='simrd ({})'.format(type(heuristic).__name__))
    return results


if __name__ == "__main__":
    logger = logging.getLogger("budget_sweep")
    logger.setLevel(logging.DEBUG)
    # due to bug on havoc, limit parallelism on high-core machines
    if os.cpu_count() > 48:
        os.environ["OMP_NUM_THREADS"] = "1"
    args = extract_params()

    ray.init(temp_dir="/tmp/ray_checkpoint", redis_password=str(uuid.uuid1()), num_cpus=os.cpu_count(),
             object_store_memory=1024 * 1024 * 1024 if os.cpu_count() < 48 else None)  # include_webui=args.debug

    key = "_".join(map(str, [args.platform, args.model_name, args.batch_size, args.input_shape]))
    log_base = remat_data_dir() / "budget_sweep" / key

    ####
    # Begin budget_sweep data collection
    ####
    model_name = args.model_name

    # load costs, and plot optionally, if platform is not flops
    logger.info(f"Loading costs")
    if args.platform == "flops":
        cost_model = None
    else:
        cost_model = CostModel(model_name, args.platform, log_base, quantization=5)
        cost_model.fit()

    # gen redis key
    if cost_model is None:
        key_list = ["flops", args.batch_size]
    else:
        key_list = [cost_model.platform, cost_model.quantization, args.batch_size]
    redis_cost_key = "_".join(map(str, key_list))

    # load model from Keras
    logger.info(f"Loading model {model_name}")
    model = get_keras_model(model_name, input_shape=args.input_shape)
    g = dfgraph_from_keras(model, batch_size=args.batch_size, cost_model=cost_model,
                           loss_cpu_cost=0, loss_ram_cost=(4 * args.batch_size))
                    
    result_dict = pickle.load((log_base / 'result_dict.pickle').open('rb'))
    simrd_eval_points = pickle.load((log_base / 'simrd_eval_points.pickle').open('rb'))

    simrd_results = []
    for heuristic in SIMRD_HEURISTICS:
        simrd_results.append(run_simrd(g, heuristic, simrd_eval_points, SIMRD_LIVENESS))

    # save simrd results and heuristics used
    pickle.dump(simrd_results, (log_base / 'simrd_results.pickle').open('wb'), \
        protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(SIMRD_HEURISTICS, (log_base / 'simrd_heuristics.pickle').open('wb'), \
        protocol=pickle.HIGHEST_PROTOCOL)
