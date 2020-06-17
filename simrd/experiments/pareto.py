import time
from typing import List

from pathos.multiprocessing import ProcessPool as Pool
from multiprocessing import cpu_count

from simrd.heuristic import *
from simrd.runtime import *
from simrd.telemetry import Telemetry

import experiments.util as util

def pareto(callback, budgets : List[float], heuristic, runtime, verbose=True, **kwargs):
  def safe_callback(rt):
    result = 'pass'
    t = time.time()
    try:
      callback(rt)
    except MemoryError:
      result = 'fail (OOM)'
    except RematExceededError:
      result = 'fail (thrashed)'
    except:
      import traceback
      traceback.print_exc()
      print(flush=True)
      raise
    total_time = time.time() - t
    rt.meta['total_time'] = total_time
    if verbose:
      print('  budget {} finished in {} seconds: {}'.format(
        rt.budget, total_time, result
      ), flush=True)

    rt._prepickle()
    return rt

  if verbose:
    print('running pareto trial for budgets: {}'.format(budgets), flush=True)

  p = Pool()

  runtimes = list(map(lambda b: runtime(b, heuristic, **kwargs), budgets))
  runtimes = p.map(safe_callback, runtimes)

  return runtimes
