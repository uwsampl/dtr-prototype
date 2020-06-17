import math

from .runtime import *

@register_runtime
class RuntimeV1(TelemetrizedRuntimeBase):
  FEATURES = TelemetrizedRuntimeBase.FEATURES.union([
    'banishing', 'last_access', 'last_access_int'
  ])
  KWARGS = {**TelemetrizedRuntimeBase.KWARGS, 'remat_limit': math.inf}
  ID = 'V1'

  def __init__(self, budget, heuristic, **kwargs):
    super().__init__(budget, heuristic, **kwargs)
    self.remat_limit = kwargs.get('remat_limit', math.inf)
    self.remat_exceeded = False
    self.storage_pool = []

  def _prepickle(self):
    super()._prepickle()
    self.storage_pool = None

  def _evictable(self, s : Storage) -> bool:
    return s in self.storage_pool

  def _make_evictable(self, s : Storage) -> bool:
    assert s.ref_int == 0, 'tried to make locked Storage evictable {}'.format(s)
    if s in self.storage_pool:
      return False
    self.storage_pool.append(s)
    return True

  def _make_unevictable(self, s : Storage) -> bool:
    if s not in self.storage_pool:
      return False
    self.storage_pool.remove(s)
    return True

  def _evict(self, s : Storage):
    assert s.ref_int == 0, 'tried to evict locked Storage {}'.format(s)
    s.material = False
    self.memory_usage -= s.size
    self._make_unevictable(s)
    # update V1 metadata
    for t in s.tensors:
      t.defined = False
      for p in t.parents:
        if s.root_id != p.storage.root_id:
          p.storage.meta['evicted_dependents'].add(t.id)
    self._T_evict(s)

  def _free(self, size : int):
    """
    Free up memory until there is `size` free space.
    """
    while self.memory_usage + size > self.budget:
      if len(self.storage_pool) == 0:
        # TODO: log anything useful gracefully; check telemetry is not messed up
        self.OOM = True
        raise MemoryError()
      evicts = self.heuristic.choose(self.storage_pool, self, telemetry=self.telemetry)
      for s in evicts:
        self._evict(s)

  def _lock(self, s : Storage):
    """
    Locks `s`, which must be material, preventing it from being evicted while
    any lock is held.
    """
    assert s.material, 'cannot lock evicted Storage {}'.format(s)
    assert not s.meta['banished'], 'cannot lock banished Storage {}'.format(s)
    if s.ref_int == 0:
      succ = self._make_unevictable(s)
      assert succ, 'unevictable Storage that is not locked {}'.format(s)
    s.ref_int += 1
    assert not self._evictable(s), 'evictable Storage after lock {}'.format(s)
    self._T_lock(s)

  def _unlock(self, s : Storage):
    """
    Releases a lock on `s`, which must be material. If this was the last lock
    (i.e. s.ref_int == 0 after), then s is put back in the pool. Note that
    pinned Storages are permanently locked.
    """
    assert s.material, 'cannot unlock evicted Storage {}'.format(s)
    assert s.ref_int > 0, 'cannot unlock Storage with no locks {}'.format(s)
    s.ref_int -= 1
    if s.ref_int == 0:
      succ = self._make_evictable(s)
      assert succ, 'Storage was evictable while locked {}'.format(s)
      self._T_unlock(s)

  def _compute(self, t : Tensor, rematerialize=True):
    for p in t.parents:
      assert p.defined, \
        'a parent Tensor is undefined {} of {}'.format(p, t)
    assert self.memory_usage + t.op.total_size <= self.budget, \
      'not enough memory to compute Tensor {}'.format(t)
  
    # Record which non-alias siblings are already material. We will use this to
    # free the doubled-computed Tensors, since operators will recompute all
    # outputs, some of which may already be in memory.
    dup_siblings = \
      list(filter(lambda u: u.defined, t.siblings))
    remat_siblings = \
      list(filter(lambda u: not u.defined, t.siblings))

    # We need to record this separately in order to make them evictable later;
    # NOTE: we include t if applicable
    remat_storages = \
      list(filter(lambda u: not u.storage.material, t.siblings + [t]))

    # 'Compute' the operator.
    self.clock += t.op.compute
    self.memory_usage += t.op.total_size  # could be 0 if t.op is a purely aliasing op

    self._T_memory_usage()

    # NOTE: it might seem like we could save more memory by first evicting the
    #       sibling tensors or something, but this might violate locks on the
    #       Storages and is hard in PyTorch, so I don't do it here.
    
    if t.meta['is_alias']:
      # If 'materializing' an alias, then it should just be defining the Tensor
      # (i.e. metadata in the PyTorch implementation). The underlying Storage
      # should ALWAYS be material at this point, via materializing the parents.
      assert t.storage.material, 'the underlying Storage of an alias was evicted {}'.format(t)

    t.storage.material = True
    t.defined = True
    self._T_compute(t, rematerialize=rematerialize, direct=True)

    # Materialize siblings
    for u in remat_siblings:
      if u.meta['is_alias']: assert u.storage.material
      u.storage.material = True
      u.defined = True
      self._T_compute(u, rematerialize=rematerialize, direct=False)

    # 'Free' the double-computed ephemeral Tensors.
    for u in dup_siblings:
      self.memory_usage -= u.op.sizes[u.index]
    
    # Make the newly materialized Storages evictable
    for u in remat_storages:
      self._make_evictable(u.storage)

    return remat_siblings, remat_storages

  def _materialize(self, t : Tensor, rematerialize=True):
    # NOTE: t.defined is False when a Tensor is an alias which has not
    #       been recomputed, even after the underlying Storage was rematerialized.
    #       This models the behavior in the PyTorch implementation, where alias
    #       Tensors are fully destroyed (rather than just the storage).
    assert not t.defined, 'cannot materialize a defined Tensor {}'.format(t)
    assert t.storage.material or t.storage.ref_int == 0, 'a locked Storage is evicted {}'.format(t.storage)
    assert not t.storage.meta['banished']

    self._T_pending(t)

    for p in t.parents:
      # Update last accessed times iff the underlying Storage is externally accessible
      if p.storage.ref_ext > 0:
        p.storage.meta['last_access_int'] = self.clock
        if not rematerialize:
          p.storage.meta['last_access'] = self.clock
      self._T_use(p, rematerialize=rematerialize)

    # TODO (MAJOR): figure out how to soundly order the rematerializations

    # get locks on all parents; rematerialize if needed
    for p in list(filter(lambda p: p.defined, t.parents)):
      self._lock(p.storage)
    for p in list(filter(lambda p: not p.defined, t.parents)):
      # NOTE: we wrap the filter in a list since lazy evaluation might cause
      #       locks to not be acquired when two parents are siblings
      if not p.defined:
        # As above, this condition may be true if siblings get rematerialized
        self._materialize(p)
      self._lock(p.storage)

    self._T_pressure(t)

    if self.memory_usage + t.op.total_size > self.budget:
      self._free(t.op.total_size)

    remat_siblings, remat_storages = self._compute(t, rematerialize=rematerialize)

    # Release locks on parents' Storages, update V1 metadata and try banish
    for p in t.parents:
      self._unlock(p.storage)
      for u in remat_siblings + [t]:
        if u.storage.root_id != p.storage.root_id:
          if u.id in p.storage.meta['evicted_dependents']:
            p.storage.meta['evicted_dependents'].remove(u.id)
      self._try_banish_V1(p.storage)

    if self.telemetry.summary['remat_compute'] > self.remat_limit:
      self.remat_exceeded = True
      raise RematExceededError('Runtime exceeded rematerialization limit.')

    # record total required memory
    self._T_bottleneck(t)

    return t

  def _try_banish_V1(self, s : Storage):
    if s.ref_ext > 0 or len(s.meta['evicted_dependents']) > 0:
      return

    assert all(map(lambda t: t.meta['ref_ext'] == 0, s.tensors)),\
      'tried to banish a Storage with live Tensors {}'.format(s)
    if s.material:
      if s.ref_int > 0:
        s.ref_int = 0
        self._T_unlock(s)
      assert s.ref_int == 0
      self._evict(s)

    # 'Detach' all Tensors in s from the computation graph, and pin dependents
    # NOTE: this needs some careful thought if we are to try and optimize like V2
    for t in s.tensors:
      for p in t.parents:
        if s.root_id != p.storage.root_id:
          # NOTE: this condition can be false due to aliasing
          if t.id in p.storage.meta['evicted_dependents']:
            p.storage.meta['evicted_dependents'].remove(t.id)
          p.children.remove(t)
      for c in t.children:
        if s.root_id != c.storage.root_id:
          c.parents.remove(t)
          self.pin(c)

    s.meta['banished'] = True
    self._T_banish(s)

  def rematerialize(self, t : Tensor):
    self._materialize(t, rematerialize=True)

  def compute(self, inputs : List[Tensor], op : Operator,
              ids : Tuple[int] = None, names : Tuple[str] = None):
    if ids == None:
      ids = list(range(self.tensor_count, self.tensor_count + op.outputs))
    op_id = self.op_count
    self.tensor_count += op.outputs
    self.op_count += 1

    tensors = Tensor.from_op(inputs, op, op_id, ids, names)
    for t in tensors:
      self.tensor_map[t.id] = t
      if 'last_access' not in t.storage.meta:
        t.storage.meta['last_access'] = -math.inf
      if 'last_access_int' not in t.storage.meta:
        t.storage.meta['last_access_int'] = -math.inf
      if 'banished' in t.storage.meta:
        assert not t.storage.meta['banished']
      else:
        t.storage.meta['banished'] = False
      # update V1 metadata for parent Storages
      if not t.meta['is_alias']:
        t.storage.meta['evicted_dependents'] = set()
      for p in inputs:
        if t.storage.root_id != p.storage.root_id:
          p.storage.meta['evicted_dependents'].add(t.id)
      self._T_new_tensor(t, op_id)

    # materialize
    self._materialize(tensors[0], rematerialize=False)
    
    for t in tensors:
      # set the last access times to avoid them being evicted immediately
      t.storage.meta['last_access'] = self.clock
      t.storage.meta['last_access_int'] = self.clock

      # finalize telemetry
      self._T_birth(t)

    return tensors

  def get(self, t : Tensor):
    assert t.storage.ref_ext > 0, \
      'cannot get a Tensor whose Storage has no external refs {}'.format(t)
    t.storage.ref_ext += 1
    t.meta['ref_ext'] += 1
    return t

  def release(self, t : Tensor):
    assert t.storage.ref_ext > 0, \
      'cannot release a Tensor whose Storage has no external refs {}'.format(t)
    t.storage.ref_ext -= 1
    t.meta['ref_ext'] -= 1
    if t.meta['ref_ext'] == 0:
      self._T_death(t)
    if t.storage.ref_ext == 0:
      self._try_banish_V1(t.storage)

  def pin(self, t : Tensor):
    assert t.storage.material, 'cannot pin Tensor with immaterial Storage {}'.format(t)
    if t.storage.pinned:
      return
    self._lock(t.storage)
    t.storage.pinned = True
    self._T_pin(t.storage)
