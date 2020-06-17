import math

from .heuristic import *
from .dtr import *
from .etc import *

@register_heuristic
class AbESizeStale(DTR):
  MARKER = '*'
  COLOR = 'green'

  def __str__(self):
    return r'$e^*$, size, staleness'

@register_heuristic
class AbESize(Heuristic):
  MARKER = 'D'
  COLOR = 'blue'
  FEATURES = set(['regions'])

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    compute = s.compute
    compute += s.meta['region'].compute + s.meta['region_rev'].compute
    denominator = s.size
    return compute / denominator if denominator > 0 else math.inf

  def __str__(self):
    return r'$e^*$, size, no staleness'

@register_heuristic
class AbEStale(Heuristic):
  MARKER = '^'
  COLOR = 'orange'
  FEATURES = set(['regions', 'last_access_int'])

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    compute = s.compute
    compute += s.meta['region'].compute + s.meta['region_rev'].compute
    denominator = Heuristic.staleness(s.meta['last_access_int'], rt.clock)
    return compute / denominator if denominator > 0 else math.inf

  def __str__(self):
    return r'$e^*$, no size, staleness'

@register_heuristic
class AbE(Heuristic):
  MARKER = 'X'
  COLOR = 'red'
  FEATURES = set(['regions'])

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    return s.compute + s.meta['region'].compute + s.meta['region_rev'].compute

  def __str__(self):
    return r'$e^*$, no size, no staleness'

@register_heuristic
class AbEqSizeStale(DTREqClass):
  MARKER = '*'
  COLOR = 'green'

  def __str__(self):
    return r'EqClass, size, staleness'

@register_heuristic
class AbEqSize(Heuristic):
  MARKER = 'D'
  COLOR = 'blue'
  FEATURES = set(['eq_class'])

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    cpi = CheckpointInfo(s.compute)
    ns = Storage.evicted_neighbors(s, rt.telemetry)
    n_cpis = set(map(lambda n: EqClassNode.get_value(n.meta['ecn']), ns))
    for n_cpi in n_cpis:
      cpi = CheckpointInfo.merge_f(cpi, n_cpi)
    denom = s.size
    return cpi.compute / denom if denom > 0 else math.inf

  def __str__(self):
    return r'EqClass, size, no staleness'

@register_heuristic
class AbEqStale(Heuristic):
  MARKER = '^'
  COLOR ='orange'
  FEATURES = set(['eq_class', 'last_access_int'])

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    cpi = CheckpointInfo(s.compute)
    ns = Storage.evicted_neighbors(s, rt.telemetry)
    n_cpis = set(map(lambda n: EqClassNode.get_value(n.meta['ecn']), ns))
    for n_cpi in n_cpis:
      cpi = CheckpointInfo.merge_f(cpi, n_cpi)
    denom = Heuristic.staleness(s.meta['last_access_int'], rt.clock)
    return cpi.compute / denom if denom > 0 else math.inf

  def __str__(self):
    return r'EqClass, no size, staleness'

@register_heuristic
class AbEq(Heuristic):
  MARKER = 'X'
  COLOR = 'red'
  FEATURES = set(['eq_class'])

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    cpi = CheckpointInfo(s.compute)
    ns = Storage.evicted_neighbors(s, rt.telemetry)
    n_cpis = set(map(lambda n: EqClassNode.get_value(n.meta['ecn']), ns))
    for n_cpi in n_cpis:
      cpi = CheckpointInfo.merge_f(cpi, n_cpi)
    return cpi.compute

  def __str__(self):
    return r'EqClass, no size, no staleness'

@register_heuristic
class AbLocalSizeStale(DTRLocal):
  MARKER = '*'
  COLOR = 'green'

  def __str__(self):
    return r'local, size, staleness'

@register_heuristic
class AbLocalSize(Heuristic):
  MARKER = 'D'
  COLOR = 'blue'

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    return s.compute / s.size if s.size > 0 else math.inf

  def __str__(self):
    return r'local, size, no staleness'

@register_heuristic
class AbLocalStale(Heuristic):
  MARKER = '^'
  COLOR = 'orange'
  FEATURES = set(['last_access_int'])

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    stale = Heuristic.staleness(s.meta['last_access_int'], rt.clock)
    return s.compute / stale if stale > 0 else math.inf

  def __str__(self):
    return r'local, no size, staleness'

@register_heuristic
class AbLocal(Heuristic):
  MARKER = 'X'
  COLOR = 'red'

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    return s.compute

  def __str__(self):
    return r'local, no size, no staleness'

@register_heuristic
class AbSizeStale(Heuristic):
  MARKER = '*'
  COLOR = 'green'
  FEATURES = set(['last_access_int'])

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    denom = s.size * Heuristic.staleness(s.meta['last_access_int'], rt.clock)
    return 1 / denom if denom > 0 else math.inf

  def __str__(self):
    return r'no cost, size, staleness'

@register_heuristic
class AbSize(LargestStorage):
  MARKER = 'D'
  COLOR = 'blue'

  def __str__(self):
    return r'no cost, size, no staleness'

@register_heuristic
class AbStale(Heuristic):
  MARKER = '^'
  COLOR = 'orange'
  FEATURES = set(['last_access_int'])

  def evaluate(self, s, rt, **kwargs):
    rt.telemetry.summary['heuristic_eval_count'] += 1
    rt.telemetry.summary['heuristic_access_count'] += 1
    stale = Heuristic.staleness(s.meta['last_access_int'], rt.clock)
    return 1 / stale if stale > 0 else math.inf

  def __str__(self):
    return r'no cost, no size, staleness'

@register_heuristic
class AbRandom(RandomStorage):
  MARKER = 'X'
  COLOR = 'red'
  TRIALS = 10

  def __str__(self):
    return r'no cost, no size, no staleness (random)'
