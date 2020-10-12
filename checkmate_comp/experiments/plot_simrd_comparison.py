import argparse
import logging
import os
import pathlib
import pickle
import shutil
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib
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

MODEL_PLATFORM = {
    'VGG16': 'p32xlarge',
    'vgg_unet': 'p32xlarge',
    'MobileNet': 'p32xlarge',
    'ResNet50': 'p32xlarge',
    # ...
}

MODEL_BATCH = {
    'VGG16': '256',
    'vgg_unet': '32',
    'MobileNet': '512',
    'ResNet50': '256',
    # ...
}

MODEL_KEY = {
    'VGG16': 'p32xlarge_VGG16_256_None',
    'vgg_unet': 'p32xlarge_vgg_unet_32_None',
    'MobileNet': 'p32xlarge_MobileNet_512_None',
    'ResNet50': 'p32xlarge_ResNet50_256_None',
    # ...
}

MODEL_INPUT_SHAPE = {
    'VGG16': '224x224',
    'vgg_unet': '416x608',
    'MobileNet': '224x224',
    'ResNet50': '224x224',
    # ...
}

MODEL_XYLIM = {
    'VGG16': [[13, 22], [0.95, 1.5]],
    'MobileNet': [[4, 48], [0.95, 1.5]],
    'vgg_unet': [[6, 40], [0.95, 1.5]],
    'ResNet50': [[8, 42], [0.95, 1.5]]
}

MODEL_TEXT = {
    'VGG16': ['VGG16 ({})'.format(MODEL_BATCH['VGG16']), MODEL_INPUT_SHAPE['VGG16']],
    'vgg_unet': ['U-Net ({})'.format(MODEL_BATCH['vgg_unet']), MODEL_INPUT_SHAPE['vgg_unet']],
    'MobileNet': ['MobileNet ({})'.format(MODEL_BATCH['MobileNet']), MODEL_INPUT_SHAPE['MobileNet']],
    'ResNet50': ['ResNet50 ({})'.format(MODEL_BATCH['ResNet50']), MODEL_INPUT_SHAPE['ResNet50']]
}

PLOT_UNIT_RAM = 1e9
PLOT_STRATEGIES = [
    SolveStrategy.CHEN_GREEDY, SolveStrategy.CHEN_GREEDY_NOAP,
    SolveStrategy.CHEN_SQRTN, SolveStrategy.CHEN_SQRTN_NOAP,
    SolveStrategy.GRIEWANK_LOGN,
    SolveStrategy.OPTIMAL_ILP_GC,
]
PLOT_HEURISTICS = ['DTR', 'DTREqClass', 'LRU']

PLOT_STRATEGY_LABELS = {
    SolveStrategy.CHEN_GREEDY: r'Chen et al. greedy',
    SolveStrategy.CHEN_SQRTN: r'Chen et al. $\sqrt{n}$',
    SolveStrategy.GRIEWANK_LOGN: r'Griewank & Walther $\log(n)$',
    SolveStrategy.CHECKPOINT_ALL: r'Checkpoint all (ideal)',
    SolveStrategy.OPTIMAL_ILP_GC: r'Checkmate (optimal ILP)',
}

PLOT_STRATEGY_ADAPT = {
    SolveStrategy.CHEN_GREEDY: '**',
    SolveStrategy.CHEN_GREEDY_NOAP: ' *',
    SolveStrategy.CHEN_SQRTN: '**',
    SolveStrategy.CHEN_SQRTN_NOAP: ' *',
}


def prefix_min_np(values: np.ndarray):
    assert values.ndim == 1
    values_min = np.copy(values)
    for i in range(1, values.shape[0]):
        values_min[i] = min(values_min[i - 1], values[i])
    return values_min


def plot_strategy(ax, results, color, marker, markersize, baseline_cpu, zorder):
    valid_data = [r.schedule_aux_data for r in results if r is not None and r.schedule_aux_data is not None]
    sorted_data = sorted(valid_data, key=lambda r: r.peak_ram)
    data_points = [(t.peak_ram / PLOT_UNIT_RAM, t.cpu * 1.0 / baseline_cpu) for t in sorted_data]
    if not len(data_points):
        return

    x, y = map(list, zip(*data_points))
    x_step, y_step = x + [10000], prefix_min_np(np.array(y + [min(y)]))
    ax.step(x_step, y_step, where='post', zorder=1, color=color)
    # Plot only the first and last points
    ax.scatter([x[0], x[-1]], [y[0], y[-1]], label="", zorder=zorder, s=markersize ** 2,
                color=color, marker=marker, alpha=0.75)

    # return first point
    return x[0], y[0]

def plot_model(model_name, fig, ax):
    log_base = remat_data_dir() / 'budget_sweep' / MODEL_KEY[model_name]
    result_dict = pickle.load((log_base / 'result_dict.pickle').open('rb'))
    simrd_results = pickle.load((log_base / 'simrd_results.pickle').open('rb'))
    simrd_heuristics = pickle.load((log_base / 'simrd_heuristics.pickle').open('rb'))
    baseline_cpu = result_dict[SolveStrategy.CHECKPOINT_ALL][0].schedule_aux_data.cpu
    baseline_memory = result_dict[SolveStrategy.CHECKPOINT_ALL][0].schedule_aux_data.peak_ram

    for solve_strategy, results in result_dict.items():
        if solve_strategy not in PLOT_STRATEGIES: continue
        color, marker, markersize = SolveStrategy.get_plot_params(solve_strategy)
        scatter_zorder = 3 if solve_strategy == SolveStrategy.CHECKPOINT_ALL_AP else 2
        pt = plot_strategy(ax, results, color, marker, markersize, baseline_cpu, scatter_zorder)

        if model_name not in CHAIN_GRAPH_MODELS and solve_strategy in PLOT_STRATEGY_ADAPT:
            icon = PLOT_STRATEGY_ADAPT[solve_strategy]
            ax.annotate(icon, xy=pt, xytext=(-14, -6), textcoords="offset points", color=color)

    # Plot simrd results
    for heuristic, results in zip(simrd_heuristics, simrd_results):
        if type(heuristic).__name__ not in PLOT_HEURISTICS:
            continue
        color, marker, markersize = heuristic.COLOR, heuristic.MARKER, matplotlib.rcParams['lines.markersize']
        plot_strategy(ax, results, color, marker, markersize, baseline_cpu, 2)

    # Plot ideal (checkpoint all)
    xlim_min, xlim_max = ax.get_xlim()
    checkpoint_all_result = result_dict[SolveStrategy.CHECKPOINT_ALL][0].schedule_aux_data
    x = baseline_memory / PLOT_UNIT_RAM
    y = 1.0
    color, marker, markersize = SolveStrategy.get_plot_params(SolveStrategy.CHECKPOINT_ALL)
    xlim_max = max(x, xlim_max)
    ax.scatter([x], [y], label="", zorder=2, color=color, marker=marker, s=markersize ** 2)
    ax.hlines(y=y, xmin=xlim_min, xmax=x, linestyles="dashed", color=color)
    ax.hlines(y=y, xmin=x, xmax=xlim_max, color=color, zorder=2)
    ax.set_xlim([xlim_min, xlim_max])

    # Plot platform memory
    ylim_min, ylim_max = ax.get_ylim()
    mem_gb = platform_memory(MODEL_PLATFORM[model_name]) / 1e9
    if xlim_min <= mem_gb <= xlim_max:
        ax.vlines(x=mem_gb, ymin=ylim_min, ymax=ylim_max, linestyles="dotted", color="b")
        ax.set_ylim([ylim_min, ylim_max])
        ax.axvspan(xlim_min, mem_gb, alpha=0.2, color='royalblue')

    xlim, ylim = MODEL_XYLIM[model_name]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Model name and settings
    title, subtitle = MODEL_TEXT[model_name]
    ax.text(0.98, 0.975, title, fontsize=14, weight='bold', ha='right', va='top', transform=ax.transAxes)
    ax.text(0.98, 0.898, subtitle, fontsize=12, ha='right', va='top', transform=ax.transAxes)

def make_legend_and_finalize(fig):
    legend_items = []
    for strategy, label in PLOT_STRATEGY_LABELS.items():
        c, m, ms = SolveStrategy.get_plot_params(strategy)
        legend_items.append(Line2D([0], [0], lw=2, label=label, color=c, marker=m, markersize=ms))
    for heuristic in PLOT_HEURISTICS:
        heuristic = eval('{}()'.format(heuristic))
        c, m, ms = heuristic.COLOR, heuristic.MARKER, matplotlib.rcParams['lines.markersize']
        label = str(heuristic)
        legend_items.append(Line2D([0], [0], lw=2, label=label, color=c, marker=m, markersize=ms))

    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Budget (GB)', fontsize=15, fontweight='bold', labelpad=8)
    plt.ylabel(r'Overhead ($\times$)', fontsize=15, fontweight='bold', labelpad=8)

    fig.legend(
        handles=legend_items, loc='upper right', bbox_to_anchor=(1.09, 0.9),
        bbox_transform=plt.gcf().transFigure, ncol=1, fancybox=False, shadow=False, frameon=True
    )

    plt.text(
        1.084, 0.38, '* Linearized adaptation\n** AP adaptation', fontsize=10,
        horizontalalignment='right', verticalalignment='top', transform=fig.transFigure
    )

if __name__ == "__main__":
    logger = logging.getLogger("budget_sweep")
    logger.setLevel(logging.DEBUG)
    log_base = remat_data_dir() / 'budget_sweep'

    sns.set()
    sns.set_style('white')

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    plot_model('VGG16', fig, ax[0])
    plot_model('MobileNet', fig, ax[1])
    plot_model('vgg_unet', fig, ax[2])
    make_legend_and_finalize(fig)

    fig.savefig(log_base / 'dtr_checkmate.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(log_base / 'dtr_checkmate.png', bbox_inches='tight', dpi=300)
