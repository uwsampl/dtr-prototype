from simrd.tensor import *
from simrd.telemetry import Telemetry

class RematExceededError(RuntimeError):
  pass

RUNTIMES = {}

def register_runtime(cls):
  RUNTIMES[cls.__name__] = cls
  return cls

class RuntimeBase:
  FEATURES  = set()
  KWARGS    = {}

  def __init__(self, budget, heuristic, **kwargs):
    self.budget = budget
    self.memory_usage = 0
    self.heuristic = heuristic
    self.OOM = False
    self.clock = 1
    self.tensor_map = {}
    self.tensor_count = 0
    self.op_count = 0
    self.meta = kwargs

    if len(heuristic.FEATURES.difference(self.FEATURES)) > 0:
      raise RuntimeError(
        'Heuristic {} requires {} but this runtime only supports {}'.format(
          str(heuristic), heuristic.FEATURES, self.FEATURES
        ))

    in_kwargs = set(kwargs.keys())
    my_kwargs = set(self.KWARGS.keys())
    diff = in_kwargs.difference(my_kwargs)
    if len(diff) > 0:
      print('WARNING: unsupported keyword arguments {}, this runtime has (with defaults) {}'.format(
        diff, self.KWARGS
      ))

  def _prepickle(self):
    self.tensor_map = None

  def compute(self, inputs : List[Tensor], op : Operator,
              ids : Tuple[int] = None, names : Tuple[str] = None):
    raise NotImplementedError

  def rematerialize(self, t : Tensor):
    raise NotImplementedError

  def get(self, t : Tensor):
    raise NotImplementedError

  def release(self, t : Tensor):
    raise NotImplementedError

  def pin(self, t : Tensor):
    raise NotImplementedError

class TelemetrizedRuntimeBase(RuntimeBase):
  FEATURES = RuntimeBase.FEATURES.union(['telemetry'])
  KWARGS   = {**RuntimeBase.KWARGS, 'stats': False, 'trace': False}

  def __init__(self, budget, heuristic, **kwargs):
    super().__init__(budget, heuristic, **kwargs)
    self.stats = kwargs.get('stats', False)
    self.trace = kwargs.get('trace', False)
    self.telemetry = Telemetry(has_stats=self.stats, has_trace=self.trace)

  def _prepickle(self):
    super()._prepickle()

  def _T_new_tensor(self, t : Tensor, op_id):
    if self.stats:
      self.telemetry.register_tensor(t, op_id)
    if t.op.name == 'constant':
      self.telemetry.summary['model_pinned_memory'] += t.op.sizes[t.index]

  def _T_birth(self, t : Tensor):
    """
    The tensor `t` has just been computed for the first time.
    """
    if not self.stats: return

    self.telemetry.set('tensor', t.id, 'birth_time', self.clock)
    if not t.meta['is_alias']:
      self.telemetry.set('storage', t.id, 'birth_time', self.clock)

  def _T_death(self, t : Tensor):
    """
    When all external references to t are lost. 
    """
    if not self.stats: return

    assert t.meta['ref_ext'] == 0
    self.telemetry.set('tensor', t.id, 'death_time', self.clock)
    if t.storage.ref_ext == 0:
      self.telemetry.set('storage', t.storage.root_id, 'death_time', self.clock)

  def _T_use(self, t : Tensor, rematerialize : bool):
    """
    The tensor `t` is going to be be used in a computation (which may or may not
    be a rematerialization). This should be called right after gaining knowledge
    of a computation involving `t`, and before any operations have been
    performed on `t` due to the computation. This is purely to log the instant
    when a Tensor has been requested for a computation.
    """
    if not self.stats: return

    key = 'remat' if rematerialize else 'model'
    for field in ['tensor', 'storage']:
      rid = t.id if field == 'tensor' else t.storage.root_id
      self.telemetry.set(field, rid, 'last_{}_use_time'.format(key), self.clock)
      self.telemetry.inc(field, rid, '{}_use_count'.format(key))
    if t.storage.meta.get('pinned', False):
      for field in ['tensor', 'storage']:
        rid = t.id if field == 'tensor' else t.storage.root_id
        self.telemetry.inc(field, rid, '{}_use_count_pinned'.format(key))

  def _T_pending(self, t : Tensor):
    """
    The Tensor `t` has just been requested to be materialized. This should be
    called before any recursive rematerializations (such as those required to
    materialize `t`). This allows recursive rematerialization chains to be tracked.
    """
    if self.trace:
      self.telemetry.trace.record('pending', self.clock, t.id)

  def _T_pressure(self, t : Tensor):
    """
    The Tensor `t` is about to be computed. This should be called after all
    parents of `t` have been obtained, and before freeing any memory to make
    space for `t` (to ensure this gets logged in the case of an OOM error).
    """
    if self.trace:
      self.telemetry.trace.pressure[self.clock] = t.op.total_size

  def _T_compute(self, t : Tensor, rematerialize : bool, direct : bool):
    """
    The Tensor `t` was just computed. This could have been a rematerialization or
    a model computation, and either direct or collateral (i.e. due to tuple op
    output). This should only be called for `t` that were previously immaterial.
    """
    # log compute
    if direct:
      self.telemetry.summary['{}_compute'.format('remat' if rematerialize else 'model')] \
        += t.op.compute

    if rematerialize and self.stats:
      key = 'direct' if direct else 'collateral'
      self.telemetry.inc('tensor', t.id, '{}_remat_count'.format(key))
      if not t.meta['is_alias']:
        self.telemetry.inc('storage', t.storage.root_id, '{}_remat_count'.format(key))
      self.telemetry.inc('operator', t.op_id, 'recompute_count')

    if self.trace:
      self.telemetry.trace.record('compute', self.clock, t.id)

  def _T_bottleneck(self, t : Tensor):
    if t.op.name != 'constant':
      req_memory = t.op.total_size
      for p in filter(lambda p: p.op.name != 'constant', t.parents):
        req_memory += p.op.sizes[p.index]
      self.telemetry.summary['bottleneck_memory'] = \
        max(self.telemetry.summary['bottleneck_memory'], req_memory)

  def _T_evict(self, s : Storage):
    """
    The Storage `s` was just evicted.
    """
    if self.stats:
      self.telemetry.inc('storage', s.root_id, 'evict_count')
    if self.trace:
      self.telemetry.trace.record('evict', self.clock, s.root_id)

  def _T_banish(self, s : Storage):
    """
    The Storage `s` has just been permanently evicted. This is not the same as
    death; for example, there is NO banishing in the eager eviction runtime.
    """
    if self.stats:
      self.telemetry.set('storage', s.root_id, 'banish_time', self.clock)
    if self.trace:
      self.telemetry.trace.record('banish', self.clock, s.root_id)

  def _T_pin(self, s : Storage):
    """
    The Storage `s` has just been pinned (can no longer be evicted by DTR).
    Do not call more than once for the same storage, as it will overwrite the time.
    """
    if self.stats:
      self.telemetry.set('storage', s.root_id, 'pin_time', self.clock)
    if self.trace:
      self.telemetry.trace.record('pin', self.clock, s.root_id)

  def _T_lock(self, s : Storage):
    """
    The Storage `s` has just been locked.
    """
    if self.trace:
      self.telemetry.trace.record('lock', self.clock, s.root_id)

  def _T_unlock(self, s : Storage):
    """
    The Storage `s` has just been unlocked, i.e. all locks have been released on
    `s` (`s.ref_int == 0`).
    """
    if self.trace:
      self.telemetry.trace.record('unlock', self.clock, s.root_id)

  def _T_memory_usage(self):
    """
    Call whenever the memory usage should be logged. This updates the max_memory
    statistic if applicable.
    """
    self.telemetry.summary['max_memory'] = \
      max(self.memory_usage, self.telemetry.summary['max_memory'])
