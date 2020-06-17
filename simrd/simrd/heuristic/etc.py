import math, random

from .heuristic import *

@register_heuristic
class LRU(Heuristic):
  FEATURES = set(['last_access_int'])
  COLOR = 'orange'
  MARKER = '^'

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    return s.meta['last_access_int']

  def __str__(self):
    return 'LRU'

@register_heuristic
class LargestStorage(Heuristic):
  COLOR = 'red'
  MARKER = 's'

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    return 1 / s.size if s.size > 0 else math.inf

  def __str__(self):
    return 'Largest Tensor'

@register_heuristic
class RandomStorage(Heuristic):
  COLOR = 'maroon'
  MARKER = 'X'
  TRIALS = 10

  def evaluate(self, s, rt, **kwargs):
    raise NotImplementedError

  def choose(self, storage_pool, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    return [storage_pool[random.randrange(0, len(storage_pool))]]

  def __str__(self):
    return 'Random Tensor (Uniform)'
