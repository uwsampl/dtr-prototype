import random

from simrd.tensor import *
from simrd.runtime import *
from simrd.telemetry import Telemetry

def run_with(n, rt, releases=True):
  forward = []
  prev_grad = None

  UNIT_OP = Operator(1, (1,), (-1,))

  # forward pass
  for i in range(n):
    parents = [] if i == 0 else [forward[i-1]]
    (ti,) = rt.compute(parents, UNIT_OP, names=(f'f{i}',))
    forward.append(ti)
  # backward pass
  if releases:
    rt.release(forward[-1])
  (prev_grad,) = rt.compute([], UNIT_OP, names=(f'b{n-1}',))
  for i in range(1, n):
    op_i = n - i - 1

    if releases:
      rt.release(forward[op_i])
    
    parents = []
    if op_i > 0:
      parents.append(forward[op_i - 1])
    parents.append(prev_grad)

    (grad,) = rt.compute(parents, UNIT_OP, names=(f'b{op_i}',))
    if releases:
      rt.release(prev_grad)
    prev_grad = grad

def run(n, B, heuristic, constructor, releases, rt_kwargs):
  rt = constructor(B, heuristic=heuristic, **rt_kwargs)

  try:
    run_with(n, rt, releases=releases)
  except (MemoryError, RematExceededError):
    pass
  except:
    import traceback
    traceback.print_exc()
    raise

  rt._prepickle()
  return rt

def chop_failures(ns, rts):
  """
  Returns `ns` and `rts` that did not OOM or thrash, by chopping all below 
  (and including) the highest OOM n, and likewise all above for the lowest 
  thrashing `n`.
  Assumes `ns` and `rts` are sorted increasing.
  """
  assert len(ns) == len(rts)
  good_ns, good_rts = [], []
  max_oom_i, min_thrash_i = -1, len(ns)
  for i, (n, rt) in enumerate(zip(ns, rts)):
    if rt.OOM and i > max_oom_i:
      max_oom_i = i
    if rt.remat_exceeded and i < min_thrash_i:
      min_trash_i = i
  return ns[max_oom_i+1:min_thrash_i], rts[max_oom_i+1:min_thrash_i]
