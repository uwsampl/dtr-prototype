from typing import List

from .tensor import Storage, Operator, Tensor

class Telemetry:
  # TODO: do we want to log Tensor/Storage remat uses after death? might be useful
  SUMMARY_STATS = [
    'model_compute', 'remat_compute', 'max_memory', 'model_const_memory', 'bottleneck_memory',
    'heuristic_access_count', 'heuristic_eval_count'
  ]
  STORAGE_STATS = [
    'root_id', 'size', 'tensor_count',
    'birth_time', 'death_time', 'last_remat_use_time', 'last_model_use_time', 'banish_time', 'pin_time',
    'remat_use_count', 'model_use_count', 'remat_use_count_pinned', 'model_use_count_pinned',
    'direct_remat_count', 'collateral_remat_count',
    'evict_count'
  ]
  OPERATOR_STATS = [
    'id', 'name', 'outputs', 'compute', 'size', 'is_aliasing', 'recompute_count'
  ]
  TENSOR_STATS = [
    'id', 'name', 'op_id', 'storage_id', 'index', 'is_alias',
    'op_name', 'size', 'compute',  # Convenience columns
    'birth_time', 'death_time', 'last_remat_use_time', 'last_model_use_time',
    'remat_use_count', 'model_use_count', 'remat_use_count_pinned', 'model_use_count_pinned',
    'direct_remat_count', 'collateral_remat_count'
  ]

  def __init__(self, has_stats=True, has_trace=True):
    self.summary = {stat : 0 for stat in Telemetry.SUMMARY_STATS}
    self.neighborhood_sizes = {}
    self.storage  = {}  # map id -> column
    self.operator = {}  # map id -> column
    self.tensor   = {}  # map id -> column
    self.trace = Trace()
    self.adj_list = {}  # map id -> List[id] (children)
    self.column_maps = {
      'storage'  : { col : Telemetry.STORAGE_STATS.index(col) for col in Telemetry.STORAGE_STATS},
      'operator' : { col : Telemetry.OPERATOR_STATS.index(col) for col in Telemetry.OPERATOR_STATS},
      'tensor'   : { col : Telemetry.TENSOR_STATS.index(col) for col in Telemetry.TENSOR_STATS}
    }
    self.has_stats = has_stats
    self.has_trace = has_trace

  def inc(self, field, rid, column, amt=1):
    field_dict = getattr(self, field)
    assert rid in field_dict
    field_dict[rid][self.column_maps[field][column]] += amt

  def set(self, field, rid, column, value):
    field_dict = getattr(self, field)
    assert rid in field_dict
    field_dict[rid][self.column_maps[field][column]] = value

  def get(self, field, rid, column):
    field_dict = getattr(self, field)
    assert rid in field_dict
    return field_dict[rid][self.column_maps[field][column]]

  def register_storage(self, s : Storage):
    root_id = s.root_id
    assert root_id not in self.storage
    self.storage[root_id] = [
      root_id, s.size, 0,
      0, 0, 0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0,
      0
    ]

  def register_operator(self, op : Operator, call_id):
    assert call_id not in self.operator
    is_aliasing = any(map(lambda i: i != -1, op.aliases))
    self.operator[call_id] = [
      call_id, op.name, op.outputs, op.compute, op.total_size, is_aliasing, 0
    ]

  def register_tensor(self, t : Tensor, call_id):
    assert t.id not in self.tensor

    # register Storage and Operator if not registered
    if t.storage.root_id not in self.storage:
      self.register_storage(t.storage)
    if call_id not in self.operator:
      self.register_operator(t.op, call_id)

    self.tensor[t.id] = [
      t.id, t.name, call_id, t.storage.root_id, t.index, t.meta['is_alias'],
      t.op.name, t.op.sizes[t.index], t.op.compute,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0
    ]

    # update storage telemetry
    self.inc('storage', t.storage.root_id, 'tensor_count')

    # update adjacency list
    for p in t.parents:
      assert p.id in self.adj_list
      # NOTE: this could be false when y = f(x,x)
      if t.id not in self.adj_list[p.id]:
        self.adj_list[p.id].append(t.id)

    assert t.id not in self.adj_list
    self.adj_list[t.id] = []

  def json(self):
    raise NotImplementedError()

# TODO: update trace to work with Storage abstraction
class Trace:
  def __init__(self, track_costs=False):
    """
    NOTE: heuristic cost tracking is currently unimplemented due to performance degradation
    """
    self.compute  = {}  # map Time -> List[Tensor id]
    self.evict    = {}  # map Time -> List[Storage id]
    self.lock     = {}  # map Time -> List[Storage id]
    self.unlock   = {}  # map Time -> List[Storage id]
    self.pending  = {}  # map Time -> List[Tensor id]
    self.pin      = {}  # map Time -> List[Storage id]
    self.banish   = {}  # map Time -> List[Storage id]
    self.pressure = {}  # map Time -> float; memory required to complete a pending op at t
    
    # NOTE: this is COSTLY; recomputed for all material tensors whenever state changes
    self.costs    = {}  # map Time -> List[Tuple[id, float]]
    self.track_costs = track_costs

    if track_costs:
      raise NotImplementedError('heuristic cost tracking is not implemented')

    self.timesteps = []  # times where something has happened
  
  def get(self, field, time):
    field = getattr(self, field)
    if time not in field:
      field[time] = []
      if time not in self.timesteps:
        self.timesteps.append(time)
    return field[time]

  def record(self, field, time, item):
    """
    Appends item to the list at `field[time]`.
    """
    self.get(field, time).append(item)

  def json(self):
    raise NotImplementedError()
