from typing import List, Union, Tuple, Mapping

class Storage:
  def __init__(self, size : int, material=False):
    """
    Invariant: tensors[0] == root
    """
    self.size = size
    self.compute = 0  # NOTE: cached
    self.material = material
    self.tensors = []
    self.ref_int = 0
    self.ref_ext = 0
    self.pinned = False
    self.root_id = None
    self.meta = {}

  def get_compute(self) -> float:
    """
    Computes and caches the compute of this Storage, in `self.compute`.
    """
    unique_ops = set()
    compute = 0
    for t in self.tensors:
      if t.op_id in unique_ops:
        continue
      compute += t.op.compute
      unique_ops.add(t.op_id)
    self.compute = compute
    return compute

  def __repr__(self):
    s_dict = dict(self.__dict__)
    s_dict['tensors'] = list(map(lambda t: t.name, self.tensors))
    return str(s_dict)

  @staticmethod
  def evicted_neighbors(s : 'Storage', tel : 'Telemetry'):
    neighbors = set()
    for t in s.tensors:
      for ps in map(lambda p: p.storage, t.parents):
        tel.summary['heuristic_access_count'] += 1
        if not ps.material and ps.root_id != s.root_id:
          neighbors.add(ps)
      for cs in map(lambda c: c.storage, t.children):
        tel.summary['heuristic_access_count'] += 1
        if not cs.material and cs.root_id != s.root_id:
          neighbors.add(cs)
    return neighbors

class Operator:
  def __init__(self, compute : float, sizes : Tuple[int], aliases : Tuple[int], name=None):
    """
    Represents an operator that produces a tuple of len(sizes) Tensors, taking
    compute time to execute. For each output o[i], sizes[i] gives the size of
    the tensor, and aliases[i] gives the index of the input tensor that o[i]
    is an alias of (OR -1 if o[i] is not an alias). Note that aliases have 0 size.

    Invariants: len(sizes) = len(aliases)
                aliases[i] != -1 ==> sizes[i] = 0
    """
    self.compute = compute
    self.sizes   = sizes
    self.aliases = aliases
    self.name    = name

    self.total_size = sum(sizes)
    self.outputs    = len(sizes)

    # check invariants
    assert len(sizes) == len(aliases)
    for i in range(self.outputs):
      if aliases[i] != -1: assert sizes[i] == 0

  def __repr__(self):
    return str(self.__dict__)

class Tensor:
  def __init__(self, parents, op : Operator, index : int, storage : Storage,
               op_id : int, tensor_id : int, name : str = None):
    self.parents  = parents.copy()
    self.siblings = []
    self.children = []
    self.index    = index
    self.op       = op
    self.storage  = storage
    self.defined  = False
    self.id       = tensor_id
    self.op_id    = op_id
    self.name     = name if name != None else 'x{}'.format(self.id)
    self.meta     = {}

    # NOTE: we assume every created Tensor has 1 external/model ref initially
    assert self not in self.storage.tensors
  
    if len(self.storage.tensors) == 0:
      self.storage.root_id = self.id

    # add self to the Tensors using storage, and bump external ref count
    self.storage.tensors.append(self)
    self.storage.ref_ext += 1
    self.storage.get_compute()  # update cached storage_compute

    # initialize helpful metadata
    self.meta['is_alias'] = self.id != self.storage.root_id
    self.meta['ref_ext'] = 1

  def __repr__(self):
    s_dict = dict(self.__dict__)
    s_dict['parents'] = list(map(lambda p: p.name, self.parents))
    s_dict['siblings'] = list(map(lambda s: s.name, self.siblings))
    s_dict['children'] = list(map(lambda c: c.name, self.children))
    return str(s_dict)

  @staticmethod
  def from_op(inputs, op : Operator, op_id : int,
              ids : Tuple[int], names : Tuple[str] = None):
    """
    Returns a tuple of sibling Tensors with inputs as parents, 'created' by the
    given op (with the given ids and names). Additionally updates the children
    of the parents, and any aliased Storages, to include the created tensors.
    This also bumps the external ref counts of any aliased storages.
    """
    outputs = []
    for i in range(op.outputs):
      name = names[i] if names != None else None
      if op.aliases[i] != -1:
        storage = inputs[op.aliases[i]].storage
      else:
        storage = Storage(op.sizes[i], material=False)
      outputs.append(Tensor(inputs, op, i, storage, op_id, ids[i], name=name))
    # set the sibling fields
    for i in range(op.outputs):
      outputs[i].siblings = outputs.copy()
      outputs[i].siblings.pop(i)
    # update parents
    for p in inputs:
      p.children.extend(outputs)

    # NOTE: additional Runtime-dependent metadata should be set/updated in
    #       the Runtime's compute() method.

    return tuple(outputs)
