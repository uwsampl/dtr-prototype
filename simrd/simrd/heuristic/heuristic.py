import math
from typing import List, Set

from ..tensor import *
from ..optimization import *
from ..telemetry import Telemetry

HEURISTICS = {}

def register_heuristic(cls):
  cls.ID = len(HEURISTICS)
  HEURISTICS[cls.__name__] = cls
  return cls

class Heuristic:
  FEATURES = set()
  COLOR = 'black'
  LINESTYLE = 'solid'
  MARKER = 'x'
  TRIALS = 1

  def __call__(self, *args, **kwargs):
    s, rt = args
    return self.evaluate(s, rt, **kwargs)

  def evaluate(self, s : Storage, rt, **kwargs):
    """
    Returns the cost for evicting Storage `s`.
    """
    raise NotImplementedError

  def choose(self, storage_pool : List[Storage], rt, **kwargs) -> List[Storage]:
    """
    From the given `storage_pool`, returns the Storage(s) to evict. By default,
    chooses a single Storage with the lowest cost. This allows for heuristics to
    incorporate batched eviction, or other features.
    """
    best_cost = math.inf
    best_storage = None
    for s in storage_pool:
      assert s.ref_int == 0 and s.material
      cost = self.evaluate(s, rt, **kwargs)
      if cost <= best_cost:
        best_cost = cost
        best_storage = s
      if cost == 0:
        break
    return [best_storage]

  @staticmethod
  def evicted_neighborhood(s : Storage, tensor_map, tel : Telemetry) -> Set[Storage]:
    """
    Returns the evicted neighborhood of `s`. This is an unoptimized/uncached
    implementation that internally creates and rebuilds `Region`s. If your
    heuristic uses evicted neighborhoods, then consider using the optimized
    runtime that maintains `s.meta['region']` and `s.meta['region_rev']`.
    """
    region = Region(s, reverse=False)
    region_rev = Region(s, reverse=True)

    region.rebuild(tel)
    region_rev.rebuild(tel)

    # NOTE: this isn't really necessary to be done here, but whatever
    nbhd = set(map(lambda tid: tensor_map[tid].storage, region.interior))
    nbhd.update(map(lambda tid: tensor_map[tid].storage, region_rev.interior))

    return nbhd

  @staticmethod
  def staleness(T, clock) -> float:
    """Returns a number representing the staleness of the timestamp `T`."""
    return clock - T

  @staticmethod
  def staleness_scaled(T, clock) -> float:
    """Returns a number in [0,1) representing the staleness of the timestamp `T`."""
    return (clock - T) / clock

  def __str__(self):
    return 'N/A (Placeholder)'
