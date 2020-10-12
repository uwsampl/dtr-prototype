import traceback

from simrd.parse import *
from simrd.runtime import *
from simrd.heuristic import Heuristic, DTR, DTREqClass

try:
  from remat.core.solvers import strategy_optimal_ilp as ilp
  from remat.core.solvers import strategy_checkpoint_last as last
  from remat.tensorflow2.extraction import dfgraph_from_keras
  from experiments.common.load_keras_model import get_keras_model, MODEL_NAMES
  from simrd.parse.checkmate import to_dfgraph, from_dfgraph
  have_checkmate = True
except:
  have_checkmate = False

have_checkmate = False

from simrd_experiments.eval.analyze import *
from simrd_experiments.eval.models import MANIFEST

def test_simrd_resnet_baseline():
  log = MANIFEST['ResNet-32 (56)']['log']
  with open(log, 'r') as f:
    g = parse_file(f, out_cond=OutputCondition.PREALLOCATE)
    print('g.meta["output_ram"] = {}'.format(g.meta['output_ram']))

  rt, result, pr = run_baseline(g.get_closure(), stats=True, trace=True)
  analysis_dir = dump_run(rt, result, pr)
  dump_csv(analysis_dir)
  analyze_memory(analysis_dir)

  g_r = rewrite_checkmate(g)
  print('  g.meta["output_ram"]   = {}'.format(g.meta['output_ram']))
  print('g_r.meta["constant_ram"] = {}'.format(g_r.meta['constant_ram']))
  rt_r, result_r, pr_r = run_baseline(g_r.get_closure(), stats=True, trace=True)
  analysis_dir_r = dump_run(rt_r, result_r, pr_r)
  dump_csv(analysis_dir_r)
  analyze_memory(analysis_dir_r)

def test_simrd_resnet_05():
  log = MANIFEST['ResNet-32 (56)']['log']
  with open(log, 'r') as f:
    g = parse_file(f, out_cond=OutputCondition.PREALLOCATE)

  rt_base, result_base, pr_base = run_baseline(g.get_closure(), stats=True, trace=True)
  rt, result, pr = run_with_callback(
    g.get_closure(), result_base, DTREqClass(), 0.5, RuntimeV2EagerOptimized,
    rt_kwargs={'stats': True, 'trace': True}
  )
  analysis_dir = dump_run(rt, result, pr)
  dump_csv(analysis_dir)
  analyze_memory(analysis_dir)

  g_r = rewrite_checkmate(g)
  print('  g.meta["output_ram"]   = {}'.format(g.meta['output_ram']))
  print('g_r.meta["constant_ram"] = {}'.format(g_r.meta['constant_ram']))
  rt_r, result_r, pr_r = run_with_callback(
    g_r.get_closure(), result_base, DTREqClass(), 0.5, RuntimeV2EagerOptimized,
    rt_kwargs={'stats': True, 'trace': True}
  )
  analysis_dir_r = dump_run(rt_r, result_r, pr_r)
  dump_csv(analysis_dir_r)
  analyze_memory(analysis_dir_r)

def test_checkmate_to_simrd_analytical_cost():
  if not have_checkmate:
    return

  test_log_filename = 'data/checkmate_simrd.log'
  batch_size = 1
  with open(test_log_filename, 'w') as test_log:
    for name in MODEL_NAMES:
      try:
        model = get_keras_model(name)
        dfg = dfgraph_from_keras(model, batch_size=batch_size, loss_cpu_cost=0, loss_ram_cost=(4 * batch_size))
        
        g = from_dfgraph(dfg)
        rt, result, pr = run_baseline(g.get_closure(), stats=True, trace=True)
        print('Baseline simrd results for {}:'.format(name), file=test_log)
        print(json.dumps(result, indent=2), file=test_log)
        print(file=test_log)
      except Exception as e:
        print('Failed for {}:'.format(name), file=test_log)
        print(traceback.format_exc(),file=test_log)
        print(file=test_log)
  print('saved Checkmate -> simrd test log to [{}]'.format(test_log_filename))

# TODO: unit tests for Checkmate -> simrd graph conversion
# TODO: implement scaffolding for getting profiled cost model

def test_graph_simple():
  g = Graph()

  op, (x,) = GOp.make(g, tuple(), 10, (5,), (-1,), 'f', ('x',), {})

  assert op.name == 'f/0'
  assert op.cost == 10
  assert not op.is_aliasing()
  assert not op.is_tuple()

  assert x.op == op
  assert x.index == 0
  assert x.name == 'x'
  assert x.storage_size == x.size() == 5
  assert x.alias() == None

  assert len(g.fwd_ops) == 1 and g.fwd_ops['f/0'] == op
  assert len(g.op_parents['f/0']) == 0
  assert len(g.tensors) == 1 and g.tensors['x'] == x
  assert g.meta['compute'] == 10

  assert g.ops_topological() == ['f/0']

def test_collapse_aliases_linear():
  graph = Graph()

  f, (x,) = GOp.make(graph, tuple(), 10, (5,), (-1,), 'f', ('x',), {})
  g, (xa,) = GOp.make(graph, (x,), 1, (0,), (0,), 'g', ('xa',), {})
  gg, (xaa,) = GOp.make(graph, (xa,), 1, (0,), (0,), 'gg', ('xaa',), {})
  h, (y,) = GOp.make(graph, (xaa,), 5, (2,), (-1,), 'h', ('y',), {})

  assert g.all_aliasing() and gg.all_aliasing()
  assert not f.is_aliasing() and not h.is_aliasing()
  assert xa.size() == xaa.size() == 0 and xa.storage_size == xaa.storage_size == 5
  assert xa.alias() == x and xaa.alias() == xa
  assert graph.meta['compute'] == 17
  assert len(graph.fwd_ops) == 4

  graph_r = rewrite_collapse_aliases(graph)
  topo = graph_r.ops_topological()

  assert len(graph_r.fwd_ops) == 2
  assert graph_r.op_children[topo[0]] == set([topo[1]])
  assert graph_r.op_parents[topo[1]] == set([topo[0]])
  assert graph_r.meta['compute'] == 15

  print()
  print('tensor rewrite map: {}'.format(graph_r.meta['tensor_map']))
  print('    op rewrite map: {}'.format(graph_r.meta['op_map']))

def test_collapse_aliases_tuple():
  graph = Graph()

  f, (x,) = GOp.make(graph, tuple(), 10, (5,), (-1,), 'f', ('x',), {})
  split, (xa, xb) = GOp.make(graph, (x,), 1, (0, 0), (0, 0), 'split', ('xa', 'xb'), {})
  h, (y,) = GOp.make(graph, (xa,), 5, (2,), (-1,), 'h', ('y',), {})
  j, (z,) = GOp.make(graph, (xb,), 4, (3,), (-1,), 't', ('z',), {})
  k, (w,) = GOp.make(graph, (xa, xb), 3, (4,), (-1,), 'k', ('w',), {})

  assert split.is_aliasing() and split.all_aliasing()
  assert not (f.is_aliasing() or h.is_aliasing() or j.is_aliasing() or k.is_aliasing())
  assert xa.size() == xb.size() == 0 and xa.storage_size == xb.storage_size == 5
  assert graph.meta['compute'] == 23
  assert len(graph.fwd_ops) == 5
  assert len(graph.tensors) == 6

  graph_r = rewrite_collapse_aliases(graph)
  topo = graph_r.ops_topological()

  assert len(graph_r.fwd_ops) == 4
  assert len(graph_r.tensors) == 4
  assert graph_r.meta['compute'] == 22

  new_x_name = graph_r.meta['tensor_map'][x.name]
  new_f_name = graph_r.meta['op_map'][f.name]
  new_op_names = [graph_r.meta['op_map'][op.name] for op in [h,j,k]]
  assert graph_r.meta['tensor_map'][xa.name] == new_x_name
  assert graph_r.meta['tensor_map'][xb.name] == new_x_name
  for new_op_name in new_op_names:
    assert graph_r.op_parents[new_op_name] == set([new_f_name])

  print()
  print('tensor rewrite map: {}'.format(graph_r.meta['tensor_map']))
  print('    op rewrite map: {}'.format(graph_r.meta['op_map']))

def test_collapse_aliases_schedule():
  graph = Graph()

  f, (x,) = GOp.make(graph, tuple(), 10, (5,), (-1,), 'f', ('x',), {})
  split, (xa, xb) = GOp.make(graph, (x,), 1, (0, 0), (0, 0), 'split', ('xa', 'xb'), {})
  h, (y,) = GOp.make(graph, (xa,), 5, (2,), (-1,), 'h', ('y',), {})
  j, (z,) = GOp.make(graph, (xb,), 4, (3,), (-1,), 't', ('z',), {})
  k, (w,) = GOp.make(graph, (xa, xb), 3, (4,), (-1,), 'k', ('w',), {})

  graph.schedule = [
    GCompute(f),
    GCompute(split), GRelease(x),
    GCompute(h), GCompute(j), GCompute(k),
    GRelease(y), GRelease(z), GRelease(w), GRelease(xa), GRelease(xb)
  ]

  schedule_pre = graph.schedule.copy()

  graph_r = rewrite_collapse_aliases(graph)
  schedule_post = graph_r.schedule.copy()
  topo = graph_r.ops_topological()

  O = {
    old_name: graph_r.ops[new_name]
      for old_name, new_name in graph_r.meta['op_map'].items()
  }
  T = {
    old_name: graph_r.tensors[new_name]
      for old_name, new_name in graph_r.meta['tensor_map'].items()
  }

  assert schedule_post == [
    GCompute(O[f.name]),
    GGet(T[x.name], pin=False), GGet(T[x.name], pin=False),
    GRelease(T[x.name]),
    GCompute(O[h.name]), GCompute(O[j.name]), GCompute(O[k.name]),
    GRelease(T[y.name]), GRelease(T[z.name]), GRelease(T[w.name]),
    GRelease(T[x.name]), GRelease(T[x.name])
  ]

  print()
  print('tensor rewrite map: {}'.format(graph_r.meta['tensor_map']))
  print('    op rewrite map: {}'.format(graph_r.meta['op_map']))
  print()
  print(' original schedule: {}'.format([str(s) for s in schedule_pre]))
  print('rewritten schedule: {}'.format([str(s) for s in schedule_post]))

def test_merge_tuples():
  graph = Graph()

  f, (x1, x2) = GOp.make(graph, tuple(), 10, (5, 3), (-1, -1), 'f', ('x1', 'x2'), {})
  h, (y,) = GOp.make(graph, (x1,), 5, (2,), (-1,), 'h', ('y',), {})
  j, (z,) = GOp.make(graph, (x2,), 4, (3,), (-1,), 't', ('z',), {})
  k, (w,) = GOp.make(graph, (x1, x2), 3, (4,), (-1,), 'k', ('w',), {})

  assert f.is_tuple()
  assert x1.size() == 5 and x2.size() == 3
  assert len(graph.tensors) == 5

  graph_r = rewrite_merge_tuples(graph)
  new_x1_name = graph_r.meta['tensor_map'][x1.name]
  new_x2_name = graph_r.meta['tensor_map'][x2.name]
  new_f_name = graph_r.meta['op_map'][f.name]
  new_op_names = [graph_r.meta['op_map'][op.name] for op in [h,j,k]]
  assert len(graph_r.tensors) == 4
  assert new_x1_name == new_x2_name
  assert graph_r.tensors[new_x1_name].size() == x1.size() + x2.size()
  assert not graph_r.ops[new_f_name].is_tuple()
  for new_op_name in new_op_names:
    assert graph_r.op_parents[new_op_name] == set([new_f_name])

  print()
  print('tensor rewrite map: {}'.format(graph_r.meta['tensor_map']))
  print('    op rewrite map: {}'.format(graph_r.meta['op_map']))

def test_merge_tuples_schedule():
  graph = Graph()

  f, (x1, x2) = GOp.make(graph, tuple(), 10, (5, 3), (-1, -1), 'f', ('x1', 'x2'), {})
  k, (w,) = GOp.make(graph, (x1, x2), 3, (4,), (-1,), 'k', ('w',), {})
  h, (y,) = GOp.make(graph, (x1,), 5, (2,), (-1,), 'h', ('y',), {})
  j, (z,) = GOp.make(graph, (x2,), 4, (3,), (-1,), 't', ('z',), {})

  graph.schedule = [
    GCompute(f),
    GCompute(k), GRelease(w),
    GCompute(h), GRelease(x1), GRelease(y),
    GCompute(j), GRelease(x2), GRelease(z)
  ]

  schedule_pre = graph.schedule.copy()

  graph_r = rewrite_merge_tuples(graph)
  schedule_post = graph_r.schedule.copy()
  topo = graph_r.ops_topological()

  O = {
    old_name: graph_r.ops[new_name]
      for old_name, new_name in graph_r.meta['op_map'].items()
  }
  T = {
    old_name: graph_r.tensors[new_name]
      for old_name, new_name in graph_r.meta['tensor_map'].items()
  }

  assert schedule_post == [
    GCompute(O[f.name]), GGet(T[x1.name], pin=False),
    GCompute(O[k.name]), GRelease(T[w.name]),
    GCompute(O[h.name]), GRelease(T[x1.name]), GRelease(T[y.name]),
    GCompute(O[j.name]), GRelease(T[x1.name]), GRelease(T[z.name])
  ]

  print()
  print('tensor rewrite map: {}'.format(graph_r.meta['tensor_map']))
  print('    op rewrite map: {}'.format(graph_r.meta['op_map']))
  print()
  print(' original schedule: {}'.format([str(s) for s in schedule_pre]))
  print('rewritten schedule: {}'.format([str(s) for s in schedule_post]))

def test_constant_elim():
  graph = Graph()
  graph.meta['no_aliases'] = True

  cf, (c,) = GOp.make(graph, tuple(), 0, (10,), (-1,), GOp.CONST_NAME, ('c',), {})
  cf2, (c2,) = GOp.make(graph, tuple(), 0, (3,), (-1,), GOp.CONST_NAME, ('c2',), {})
  g, (x,) = GOp.make(graph, (c, c2), 5, (5,), (-1,), 'g', ('x',), {})

  graph.schedule = [
    GCompute(cf), GCompute(cf2), GCompute(g), GRelease(x)
  ]

  assert len(graph.fwd_ops) == 3
  assert graph.op_parents[g.name] == set([cf.name, cf2.name])

  graph_r = rewrite_constant_elim(graph)
  assert len(graph_r.fwd_ops) == 1
  assert graph_r.meta['constant_ram'] == 13
  assert graph_r.op_parents[graph_r.meta['op_map'][g.name]] == set()

  assert graph_r.schedule == [
    GCompute(graph_r.ops[graph_r.meta['op_map'][g.name]]),
    GRelease(graph_r.tensors[graph_r.meta['tensor_map'][x.name]])
  ]
