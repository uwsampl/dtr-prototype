from remat.core.dfgraph import DFGraph
from remat.core.schedule import ScheduledResult, SchedulerAuxData
from remat.core.enum_strategy import SolveStrategy
from remat.core.utils.timer import Timer

from simrd.heuristic import Heuristic
from simrd.runtime import TelemetrizedRuntimeBase, RuntimeV1, RematExceededError
from simrd.parse.checkmate import from_dfgraph

import math
import numpy as np

def get_simrd_base_compute(callback):
  rt = RuntimeV1(math.inf, Heuristic(), stats=False, trace=False)
  callback(rt)
  return rt.clock - 1

def solve_simrd(g: DFGraph, budget: int, heuristic: Heuristic, runtime: TelemetrizedRuntimeBase,
                thrash: float, liveness=True):
  simrd_g = from_dfgraph(g, liveness_analysis=liveness)
  callback = simrd_g.get_closure()
  base_compute = sum(g.cost_cpu.values())
  assert np.isclose(base_compute, get_simrd_base_compute(callback))
  remat_limit  = base_compute * (thrash - 1)
  rt = runtime(budget=budget, heuristic=heuristic, remat_limit=remat_limit)

  try:
    with Timer("solve_simrd") as timer_solve:
      callback(rt)
      assert rt.clock - 1 >= base_compute
      assert np.isclose(rt.clock - 1, rt.telemetry.summary['model_compute'] + rt.telemetry.summary['remat_compute'])
      feasible = True
  except (MemoryError, RematExceededError):
    feasible = False

  # TODO: build schedule from execution

  if feasible:
    aux_data = SchedulerAuxData(
      R=None, S=None,
      cpu=rt.clock - 1,
      peak_ram=rt.telemetry.summary['max_memory'] + g.cost_ram_fixed,
      activation_ram=rt.telemetry.summary['max_memory'],
      mem_grid=None, mem_timeline=None,
      schedule_time_s=None
    )
  else:
    aux_data = None

  return ScheduledResult(
    solve_strategy=SolveStrategy.SIMRD,
    solver_budget=budget,
    feasible=feasible,
    schedule=None,
    schedule_aux_data=aux_data,
    ilp_aux_data=None,
    solve_time_s=timer_solve.elapsed
  )
