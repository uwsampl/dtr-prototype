from remat.core.dfgraph import DFGraph

from .graph import *

def to_dfgraph(g : Graph) -> DFGraph:
  assert all([g.meta['no_aliases'], g.meta['no_tuples'], g.meta['no_constants']])

  op_map = {}
  def map_op(name : str) -> int:
    if name not in op_map:
      op_map[name] = len(op_map)
    return op_map[name]

  args = {}
  for t in g.ops.keys():
    parents = sorted([map_op(p) for p in g.op_parents[t]])
    args[map_op(t)] = parents

  v = op_map.values()
  cost_cpu = {op_map[t]: g.ops[t].cost for t in g.ops.keys()}
  cost_ram = {op_map[t]: sum(g.ops[t].size) for t in g.ops.keys()}
  cost_ram_parameters = g.meta['constant_ram']

  return DFGraph(
    args,
    v,
    None,  # TODO: vfwd_map
    None,  # TODO: vloss
    cost_cpu,
    cost_ram,
    None,
    cost_ram_parameters
  )

def from_dfgraph(dfg : DFGraph, liveness_analysis=True) -> Graph:
  g = Graph()
  g.meta['cost_ram_parameters'] = dfg.cost_ram_parameters

  topo = dfgraph_topological(dfg)
  op_map = {}
  tensor_map = {}

  # add operators in topological order
  for v in topo:
    args = tuple([tensor_map[u] for u in dfg.args[v]])
    op, (tensor,) = GOp.make(
      g,
      args,
      float(dfg.cost_cpu[v]),
      (int(dfg.cost_ram[v]),),
      (-1,),
      'f{}'.format(v),
      ('x{}'.format(v),),
      {},
      make_uname=False
    )
    op_map[v] = op
    tensor_map[v] = tensor

  g.meta['op_map'] = op_map
  g.meta['tensor_map'] = tensor_map

  # create schedule using topo ordering + liveness analysis; TODO: improve?
  if liveness_analysis:
    last_used, can_free = analyze_liveness(topo, dfg)
  schedule = []
  for (i, v) in enumerate(topo):
    schedule.append(GCompute(op_map[v]))
    if liveness_analysis:
      for u in can_free[i]:
        schedule.append(GRelease(tensor_map[u]))
  g.schedule = schedule

  return g

def dfgraph_topological(dfg : DFGraph) -> List[int]:
  visited = {v : False for v in dfg.v}
  stack = []
  def visit(v):
    visited[v] = True
    for u in dfg.successors(v):
      if not visited[u]:
        visit(u)
    stack.insert(0, v)
  for v in dfg.v:
    if not visited[v]:
      visit(v)
  return stack

def analyze_liveness(ordering : List[int], dfg : DFGraph) -> Mapping[int, int]:
  """
  Map v -> index in ordering where v is last used (or computed), and
  Map index -> set of v which can be freed after index.
  """
  last_used = {}
  for (i, v) in enumerate(ordering):
    for u in dfg.args[v]:
      last_used[u] = i
    last_used[v] = i

  can_free = defaultdict(set)
  for v, i in last_used.items():
    can_free[i].add(v)

  return last_used, can_free
