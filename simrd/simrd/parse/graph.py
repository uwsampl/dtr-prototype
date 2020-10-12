import attr
from attr import attrib, s
from typing import Tuple, List, Optional, Callable, Mapping, Union, Set
from collections import defaultdict

from ..tensor import Operator

@attr.s(auto_attribs=True)
class GOp:
  cost    : float
  size    : Tuple[int]
  alias   : Tuple[int]
  args    : Tuple['GTensor']
  result  : Tuple['GTensor'] 
  name    : str
  meta    : dict

  def __attrs_post_init__(self):
    assert len(self.size) == len(self.alias) == len(self.result)
    for i in range(len(self.size)):
      assert self.alias[i] == -1 or self.size[i] == 0

  def is_aliasing(self) -> bool:
    return any([a >= 0 for a in self.alias])

  def all_aliasing(self) -> bool:
    return all([a >= 0 for a in self.alias])

  def is_tuple(self) -> bool:
    return len(self.result) > 1

  def __str__(self): return self.name

  @staticmethod
  def make(g : 'Graph',
           args : Tuple['GTensor'],
           cost : float,
           size : Tuple[int],
           alias : Tuple[int],
           name : str,
           res_names : Tuple[str],
           meta : dict,
           make_uname : bool = True) -> ('GOp', Tuple['GTensor']):
    assert len(size) == len(alias) == len(res_names)
    uname = '{}/{}'.format(name, g._next_id()) if make_uname else name
    result = tuple([GTensor(None, i, res_names[i], None) for i in range(len(res_names))])
    op = GOp(cost, size, alias, args, result, uname, meta)
    for r in result:
      r.op = op
      r.storage_size = r.size() if not r.alias() else r.alias().storage_size
      assert r.storage_size is not None
    g.add_op(op)
    return op, result

GOp.CONST_NAME = 'constant'

@attr.s(auto_attribs=True)
class GTensor:
  op : 'GOp'
  index : int
  name : str
  storage_size : int
  meta : dict = attrib(factory=dict)

  def size(self) -> int:
    return self.op.size[self.index]

  def alias(self) -> Optional['GTensor']:
    a = self.op.alias[self.index]
    return self.op.args[a] if a >= 0 else None

  def __str__(self): return self.name

@attr.s(auto_attribs=True)
class GCompute:
  op : 'GOp'

  def __str__(self):
    return '({},)=Compute({})'.format(
      ','.join([r.name for r in self.op.result]), 
      self.op.name
    )

@attr.s(auto_attribs=True)
class GGet:
  tensor : 'GTensor'
  pin : bool

  def __str__(self):
    op = 'Pin' if self.pin else 'Get'
    return '{}({})'.format(op, self.tensor.name)

@attr.s(auto_attribs=True)
class GRelease:
  tensor : 'GTensor'

  def __str__(self):
    return 'Release({})'.format(self.tensor.name)

class Graph:
  def __init__(self):
    self._id : int = 0
    self.schedule : List[Union['GCompute', 'GGet', 'GRelease']] = []
    self.ops      : Mapping[str, 'GOp'] = {}
    self.fwd_ops  : Mapping[str, 'GOp'] = {}
    self.bwd_ops  : Mapping[str, 'GOp'] = {}
    self.tensors  : Mapping[str, 'GTensor'] = {}
    self.op_children : Mapping[str, Set[str]] = defaultdict(set)
    self.op_parents  : Mapping[str, Set[str]] = defaultdict(set)
    self.meta = {
      'compute': 0
    }

  def _next_id(self) -> int:
    i = self._id
    self._id += 1
    return i

  def add_op(self, op : 'GOp') -> None:
    assert op.name not in self.ops
    self.ops[op.name] = op
    if op.meta.get('bwd', False):
      self.bwd_ops[op.name] = op
    else:
      self.fwd_ops[op.name] = op
    for ti in op.args:
      assert ti.name in self.tensors
    op_parents = set([ti.op.name for ti in op.args])
    for ps in op_parents:
      self.op_children[ps].add(op.name)
    self.op_parents[op.name] = op_parents
    for to in op.result:
      assert to.name not in self.tensors
      self.tensors[to.name] = to
    self.meta['compute'] += op.cost

  # returns op names, not ops
  def ops_topological(self) -> List[str]:
    visited = {v : False for v in self.ops}
    stack   = []
    def visit(v):
      visited[v] = True
      for u in self.op_children[v]:
        if not visited[u]:
          visit(u)
      stack.insert(0, v)
    for v in self.ops:
      if not visited[v]:
        visit(v)
    return stack

  def get_closure(self) -> Callable[['Runtime'], None]:
    def f(rt):
      tensor_map = {}
      for cmd in self.schedule:
        if isinstance(cmd, GCompute):
          # TODO: add a rematerialize cmd? this assumes once-compute only
          for x in cmd.op.args:
            assert x.name in tensor_map
          args = [tensor_map[x.name] for x in cmd.op.args]
          rt_op = Operator(
            cmd.op.cost,
            cmd.op.size,
            cmd.op.alias,
            cmd.op.name
          )
          res = rt.compute(args, rt_op, names=tuple([o.name for o in cmd.op.result]))
          for i, r in enumerate(res):
            assert cmd.op.result[i].name not in tensor_map
            tensor_map[cmd.op.result[i].name] = r
        elif isinstance(cmd, GGet):
          assert cmd.tensor.name in tensor_map
          t = tensor_map[cmd.tensor.name]
          if cmd.pin:
            if not t.defined:
              rt.rematerialize(t)
            assert t.defined
            rt.pin(t)
          else:
            rt.get(t)
        elif isinstance(cmd, GRelease):
          assert cmd.tensor.name in tensor_map
          rt.release(tensor_map[cmd.tensor.name])
    return f

def rewrite_collapse_aliases(g : 'Graph') -> 'Graph':
  g_r = Graph()
  g_r.meta = g.meta.copy()
  g_r.meta['compute'] = 0

  ops_topological = g.ops_topological()
  # maps old -> new
  tensor_map : Mapping[str, 'GTensor'] = {}
  op_map : Mapping[str, 'GOp'] = {}

  for op_name in ops_topological:
    op = g.ops[op_name]
    if op.is_aliasing():
      if not op.all_aliasing():
        raise RuntimeError(
          'cannot collapse aliases, {} is not all aliasing'
          .format(op)
        )
      for r in op.result:
        tensor_map[r.name] = tensor_map[r.alias().name]
    else:
      # keep operator
      args = [tensor_map[x.name] for x in op.args]
      op_new, res = GOp.make(
        g_r, args, op.cost, op.size, op.alias,
        op.name, tuple([o.name for o in op.result]), op.meta,
        make_uname=False
      )
      for r in res:
        tensor_map[r.name] = r
      op_map[op.name] = op_new

  # rewrite schedule
  for cmd in g.schedule:
    if isinstance(cmd, GCompute):
      if cmd.op.name in op_map:
        g_r.schedule.append(GCompute(op_map[cmd.op.name]))
      else:
        # aliasing op; increase refcount
        for r in cmd.op.result:
          g_r.schedule.append(GGet(tensor_map[r.name], pin=False))
    elif isinstance(cmd, GGet):
      g_r.schedule.append(GGet(tensor_map[cmd.tensor.name], pin=cmd.pin))
    elif isinstance(cmd, GRelease):
      g_r.schedule.append(GRelease(tensor_map[cmd.tensor.name]))

  g_r.meta['no_aliases'] = True
  g_r.meta['tensor_map'] = {old: new.name for old, new in tensor_map.items()}
  g_r.meta['op_map'] = {old: new.name for old, new in op_map.items()}

  return g_r

def rewrite_merge_tuples(g : 'Graph') -> 'Graph':
  g_r = Graph()
  g_r.meta = g.meta.copy()
  g_r.meta['compute'] = 0

  ops_topological = g.ops_topological()
  # maps old -> new
  tensor_map : Mapping[str, 'GTensor'] = {}
  op_map : Mapping[str, 'GOp'] = {}

  for op_name in ops_topological:
    op = g.ops[op_name]
    assert not op.is_aliasing()
    if op.is_tuple():
      args = tuple([tensor_map[x.name] for x in op.args])
      op_new, res = GOp.make(
        g_r, args, op.cost, (sum(op.size),), (-1,),
        op.name, ('+'.join([o.name for o in op.result]),), op.meta,
        make_uname=False
      )
      for r in op.result:
        tensor_map[r.name] = res[0]
      op_map[op.name] = op_new
    else:
      # keep
      args = [tensor_map[x.name] for x in op.args]
      op_new, res = GOp.make(
        g_r, args, op.cost, op.size, op.alias,
        op.name, (op.result[0].name,), op.meta,
        make_uname=False
      )
      tensor_map[res[0].name] = res[0]
      op_map[op.name] = op_new

  for cmd in g.schedule:
    if isinstance(cmd, GCompute):
      op_new = op_map[cmd.op.name]
      g_r.schedule.append(GCompute(op_new))
      # need to get more refs for each missing tuple output
      for _ in range(len(cmd.op.result) - 1):
        g_r.schedule.append(GGet(op_new.result[0], pin=False))
    elif isinstance(cmd, GGet):
      g_r.schedule.append(GGet(tensor_map[cmd.tensor.name], pin=cmd.pin))
    elif isinstance(cmd, GRelease):
      g_r.schedule.append(GRelease(tensor_map[cmd.tensor.name]))

  g_r.meta['no_tuples'] = True
  g_r.meta['tensor_map'] = {old: new.name for old, new in tensor_map.items()}
  g_r.meta['op_map'] = {old: new.name for old, new in op_map.items()}

  return g_r

def rewrite_constant_elim(g : 'Graph') -> 'Graph':
  if not g.meta.get('no_aliases', False):
    raise RuntimeError('cannot eliminate constants, input graph may have aliases')

  g_r = Graph()
  g_r.meta = g.meta.copy()
  compute_pre = g_r.meta['compute']
  g_r.meta['compute'] = 0
  g_r.meta['constant_ram'] = 0

  ops_topological = g.ops_topological()
  # maps old -> new
  tensor_map : Mapping[str, 'GTensor'] = {}
  op_map : Mapping[str, 'GOp'] = {}

  for op_name in ops_topological:
    op = g.ops[op_name]
    if op_name.split('/')[0] == GOp.CONST_NAME:
      args = [tensor_map[x.name] for x in op.args]
      assert len(args) == 0
      g_r.meta['constant_ram'] += sum(op.size)
    else:
      # keep operator
      args = [tensor_map[x.name] for x in op.args if x.name in tensor_map]
      op_new, res = GOp.make(
        g_r, args, op.cost, op.size, op.alias,
        op.name, tuple([o.name for o in op.result]), op.meta,
        make_uname=False
      )
      for r in res:
        tensor_map[r.name] = r
      op_map[op.name] = op_new

  for cmd in g.schedule:
    if isinstance(cmd, GCompute):
      if cmd.op.name in op_map:
        op_new = op_map[cmd.op.name]
        g_r.schedule.append(GCompute(op_new))
    elif isinstance(cmd, GGet):
      if cmd.tensor.name in tensor_map:
        g_r.schedule.append(GGet(tensor_map[cmd.tensor.name], pin=cmd.pin))
    elif isinstance(cmd, GRelease):
      if cmd.tensor.name in tensor_map:
        g_r.schedule.append(GRelease(tensor_map[cmd.tensor.name]))

  g_r.meta['no_constants'] = True
  g_r.meta['tensor_map'] = {old: new.name for old, new in tensor_map.items()}
  g_r.meta['op_map'] = {old: new.name for old, new in op_map.items()}

  assert compute_pre == g_r.meta['compute']

  return g_r

def rewrite_checkmate(g : 'Graph') -> 'Graph':
  g_r = rewrite_collapse_aliases(g)
  g_r = rewrite_merge_tuples(g_r)
  g_r = rewrite_constant_elim(g_r)
  return g_r
