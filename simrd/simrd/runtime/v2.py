import math

from .runtime import *
from ..optimization import *

@register_runtime
class RuntimeV2(TelemetrizedRuntimeBase):
  FEATURES = TelemetrizedRuntimeBase.FEATURES.union([
    'last_evict', 'last_access', 'last_access_int'
  ])
  KWARGS = {
    **TelemetrizedRuntimeBase.KWARGS,
    'remat_limit': math.inf
  }
  ID = 'V2'

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
    s.meta['last_evict'] = self.clock
    self.memory_usage -= s.size
    self._make_unevictable(s)
    for t in s.tensors:
      t.defined = False
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
      assert p.storage.material, \
        'a parent Tensor has immaterial storage {} of {}'.format(p, t)
    assert self.memory_usage + t.op.total_size <= self.budget, \
      'not enough memory to compute Tensor {}'.format(t)
    assert not t.defined
  
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

    self._T_pending(t)

    for p in t.parents:
      # Update last accessed times iff the underlying Storage is externally accessible
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

    self._compute(t, rematerialize=rematerialize)

    # Release locks on parents' Storages
    for p in t.parents:
      self._unlock(p.storage)

    if self.telemetry.summary['remat_compute'] > self.remat_limit:
      self.remat_exceeded = True
      raise RematExceededError('Runtime exceeded rematerialization limit.')

    # record total required memory
    self._T_bottleneck(t)

    return t

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

  def pin(self, t : Tensor):
    assert t.storage.material, 'cannot pin Tensor with immaterial Storage {}'.format(t)
    if t.storage.pinned:
      return
    self._lock(t.storage)
    t.storage.pinned = True
    self._T_pin(t.storage)

@register_runtime
class RuntimeV2Eager(RuntimeV2):
  FEATURES = RuntimeV2.FEATURES.union([
    'eager_evict'
  ])

  # this implements eager eviction
  def release(self, t):
    super().release(t)
    s = t.storage
    if s.ref_ext == 0 and s.material and not s.pinned:
      assert s.ref_int == 0
      self._evict(s)

@register_runtime
class RuntimeV2Optimized(RuntimeV2):
  FEATURES = RuntimeV2.FEATURES.union([
    'regions', 'eq_class'
  ])

  def _evict(self, s : Storage):
    super()._evict(s)

    # TODO: profile, if slow then make it a static boolean
    if 'regions' in self.heuristic.FEATURES:
      # Update regions
      for ps_id in s.meta['region_rev'].frontier:
        ps = self.tensor_map[ps_id].storage
        ps.meta['region'].absorb(s, self.tensor_map, self.telemetry)
      for cs_id in s.meta['region'].frontier:
        cs = self.tensor_map[cs_id].storage
        cs.meta['region_rev'].absorb(s, self.tensor_map, self.telemetry)
      s.meta['region'].clear()
      s.meta['region_rev'].clear()

    if 'eq_class' in self.heuristic.FEATURES:
      # Make an EqClassNode for s and merge with DISTINCT neighboring EqClasses
      assert s.meta.get('ecn', None) is None
      s.meta['ecn'] = EqClassNode(CheckpointInfo(s.compute), tel=self.telemetry)
      neighbors = Storage.evicted_neighbors(s, self.telemetry)
      # TODO: optimize merge order?
      for n in neighbors:
        assert n.meta.get('ecn', None) is not None
        # Merge will only merge the same EqClass once
        EqClassNode.merge(CheckpointInfo.merge_f, n.meta['ecn'], s.meta['ecn'])

  def _compute(self, t : Tensor, rematerialize=True):
    remat_siblings, remat_storages = super()._compute(t, rematerialize=rematerialize)

    if 'regions' in self.heuristic.FEATURES:
      # Rebuild regions
      affected_regions = set()
      affected_regions_rev = set()
      for u in remat_storages:
        us = u.storage
        us.meta['region'].rebuild(self.telemetry)
        us.meta['region_rev'].rebuild(self.telemetry)
        affected_regions.update(us.meta['region_rev'].frontier)
        affected_regions_rev.update(us.meta['region'].frontier)
      for s_id in affected_regions:
        self.tensor_map[s_id].storage.meta['region'].rebuild(self.telemetry)
      for s_id in affected_regions_rev:
        self.tensor_map[s_id].storage.meta['region_rev'].rebuild(self.telemetry)

    if 'eq_class' in self.heuristic.FEATURES:
      for s in map(lambda u: u.storage, remat_storages):
        assert s.meta.get('ecn', None) is not None
        cpi_pre  = EqClassNode.get_value(s.meta['ecn'])
        cpi_post = CheckpointInfo(cpi_pre.compute - s.compute)
        EqClassNode.set_value(s.meta['ecn'], cpi_post)
        s.meta['ecn'] = None

    return remat_siblings, remat_storages

  # Overload compute() to add regions as necessary
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
      if 'regions' in self.heuristic.FEATURES:
        if 'region' not in t.storage.meta:
          t.storage.meta['region'] = Region(t.storage, reverse=False)
        if 'region_rev' not in t.storage.meta:
          t.storage.meta['region_rev'] = Region(t.storage, reverse=True)
      if 'eq_class' in self.heuristic.FEATURES and 'ecn' not in t.storage.meta:
        t.storage.meta['ecn'] = EqClassNode(CheckpointInfo(0), tel=self.telemetry)
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

@register_runtime
class RuntimeV2EagerOptimized(RuntimeV2Optimized):
  FEATURES = RuntimeV2Optimized.FEATURES.union([
    'eager_evict'
  ])

  # this implements eager eviction
  def release(self, t):
    super().release(t)
    s = t.storage
    if s.ref_ext == 0 and s.material and not s.pinned:
      assert s.ref_int == 0
      self._evict(s)
