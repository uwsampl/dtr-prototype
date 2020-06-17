import math

from ..tensor import *
from ..optimization import *
from .heuristic import *

@register_heuristic
class DTR(Heuristic):
  FEATURES = set(['regions', 'last_access_int'])
  COLOR = 'green'
  MARKER = '*'

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    compute = s.compute
    compute += s.meta['region'].compute + s.meta['region_rev'].compute
    denominator = s.size * Heuristic.staleness(s.meta['last_access_int'], rt.clock)
    return compute / denominator if denominator > 0 else math.inf

  def __str__(self):
    return 'DTR-Full'

@register_heuristic
class DTREqClass(Heuristic):
  FEATURES = set(['eq_class', 'last_access_int'])
  COLOR = 'royalblue'
  MARKER = 'D'

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    cpi = CheckpointInfo(s.compute)
    # NOTE: we only want unique CPIs, don't overcount
    ns = Storage.evicted_neighbors(s, rt.telemetry)
    n_cpis = set(map(lambda n: EqClassNode.get_value(n.meta['ecn']), ns))
    for n_cpi in n_cpis:
      cpi = CheckpointInfo.merge_f(cpi, n_cpi)
    denom = s.size * Heuristic.staleness(s.meta['last_access_int'], rt.clock)
    return cpi.compute / denom if denom > 0 else math.inf

  def __str__(self):
    return 'DTR-EqClass'

@register_heuristic
class DTRLocal(Heuristic):
  FEATURES = set(['last_access_int'])
  COLOR = 'rebeccapurple'
  MARKER = 'o'

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    denom = s.size * Heuristic.staleness(s.meta['last_access_int'], rt.clock)
    return s.compute / denom if denom > 0 else math.inf

  def __str__(self):
    return 'DTR-Local'

@register_heuristic
class DTRUnopt(Heuristic):
  FEATURES = set(['last_access_int'])
  COLOR = 'green'
  MARKER = '*'

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    nbhd = Heuristic.evicted_neighborhood(s, rt.tensor_map, rt.telemetry)
    compute = s.compute
    last_access = s.meta['last_access_int']
    for u in nbhd:
      compute += u.compute
      last_access = max(last_access, u.meta['last_access_int'])
    denom = s.size * Heuristic.staleness(last_access, rt.clock)
    return compute / denom if denom > 0 else math.inf

  def __str__(self):
    return 'DTR-Full (Unoptimized)'
