import json
import time
import glob
from datetime import datetime
from pathos.multiprocessing import ProcessPool as Pool
import matplotlib.pyplot as plt

from simrd.heuristic import *
from simrd.runtime import *

from experiments.bounds import *
import experiments.util as util

from experiments.uniform_linear.run import run, chop_failures

"""
Experiments to evaluate the best-case asymptotic behavior of DTR using various
heuristics, by running the simulator on different memory/heuristic pairings.
These experiments do not capture any runtime overhead (hence 'best case'
performance).
"""

ASYMPTOTICS_MOD = 'uniform_linear/asymptotics'

def run_asymptotics(base, ns, heuristic, bound, runtime, releases=True, **kwargs):
  config = {
    'ns': ns,
    'heuristic': str(heuristic),
    'heuristic_features': list(heuristic.FEATURES),
    'memory': str(bound),
    'releases': releases,
    'runtime': runtime.ID,
    'runtime_features': list(runtime.FEATURES),
    'kwargs': kwargs
  }

  p = Pool()

  print('generating asymptotics data for config: {}...'.format(json.dumps(config, indent=2)))

  args = []
  for n in ns:
    args.append([n, bound(n), heuristic, runtime, releases, kwargs])

  t = time.time()
  rts = p.map(run, *zip(*args))
  t = time.time() - t
  succ_ns, succ_rts = chop_failures(ns, rts)
  print('  - succeeded between n={} and n={}'.format(succ_ns[0], succ_ns[-1]))
  print('  done, took {} seconds.'.format(t))
  results = {
    'layers': succ_ns,
    'computes': list(map(lambda rt: rt.telemetry.summary['remat_compute'], rts)),
    'had_OOM': ns[0] != succ_ns[0],
    'had_thrash': ns[-1] != succ_ns[-1]
  }

  date_str = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
  base_mod = ASYMPTOTICS_MOD + '/' + base
  out_file = '{}-{}-{}.json'.format(date_str, heuristic.ID, bound.ID)
  util.ensure_output_path(base_mod)
  out_path = util.get_output_path(base_mod, out_file)
  with open(out_path, 'w') as out_f:
    out_f.write(json.dumps({'config': config, 'results': results}, indent=2))
  print('-> done, saved to "{}"'.format(out_path))

def plot_treeverse(ns):
  def treeverse(n):
    if n == 0:
        return 0
    if n == 1:
        return 1 + 1 # forward then backward
    else:
        m = n // 2
        assert m * 2 == n
        return m + 2 * treeverse(m)
  tv_ns = sorted(set(map(lambda n: 2 ** (n.bit_length() - 1), ns)))
  tv_rs = [treeverse(n) - n for n in tv_ns]
  plt.plot(tv_ns, tv_rs, \
    label=r'TREEVERSE ($B=\log_2(n)$, theoretical)', color='black', linestyle='--', alpha=0.8)

def plot_tq(ns):
  plt.plot(ns, ns, \
    label=r'Chen et al. ($2\sqrt{n}$, theoretical)', color='black', linestyle='--', alpha=0.8)

def run_runtime_comparison(base, ns, heuristic, bound):
  for eager in [False, True]:
    runtime = RuntimeV2EagerOptimized if eager else RuntimeV2Optimized
    run_asymptotics(base, ns, heuristic, bound, runtime)

def run_heuristic_comparison(base, ns, heuristics, bound, runtime, releases=True, **kwargs):
  for heuristic in heuristics:
    run_asymptotics(base, ns, heuristic, bound, runtime, releases=releases, **kwargs)

def plot_runtime_comparison(base, heuristic, bound, out_file):
  """Compare different runtime settings on the same budget and heuristic."""
  base_dir = util.get_output_path(ASYMPTOTICS_MOD + '/' + base, '')
  results = []
  paths = glob.glob(base_dir + '*.json')
  for result_path in paths:
    js = None
    with open(result_path, 'r') as jf:
      js = json.loads(jf.read())
    assert js is not None
    # if js['config']['runtime'] != 'V2': continue
    if js['config']['heuristic'] != str(heuristic) and \
      js['config']['heuristic'] != str(heuristic) + ' (Unoptimized)':
      print('skipping "{}", since {} != {}'.format(result_path, js['config']['heuristic'], str(heuristic)))
      continue
    if js['config']['memory'] != str(bound):
      print('skipping "{}", since {} != {}'.format(result_path, js['config']['memory'], str(bound)))
      continue
    assert len(results) == 0 or results[-1]['config']['releases'] == js['config']['releases']
    assert len(results) == 0 or results[-1]['config']['kwargs'] == js['config']['kwargs']
    results.append(js)

  for res in results:
    v1_banishing = res['config']['runtime'] == 'V1'
    if not v1_banishing:
      eager_evict = 'eager_evict' in res['config']['runtime_features']
      if eager_evict:
        runtime_label = 'Eager eviction'
      else:
        runtime_label = 'No eager eviction'
    else:
      runtime_label = 'Banishing'
    l = plt.plot(res['results']['layers'], res['results']['computes'], label=runtime_label, alpha=0.7, marker='X')
    l_color = l[0].get_color()
    if res['results']['had_OOM']:
      plt.axvline(x=res['results']['layers'][0], color=l_color, linestyle='dotted', \
        alpha=0.3, linewidth=3)
    if res['results']['had_thrash']:
      plt.axvline(x=res['results']['layers'][-1], color=l_color, linestyle='--', \
        alpha=0.3, linewidth=3)
  if isinstance(bound, Log2Bound):
    plot_treeverse(results[0]['config']['ns'])
  else:
    plot_tq(results[0]['config']['ns'])
  plt.grid(True)
  plt.xlabel(r'Number of Layers $n$')
  plt.ylabel(r'Additional Compute')
  plt.legend()
  plt.title('Layers vs. Compute Overhead\n{} Heuristic, {} Memory'.format(str(heuristic), str(bound)))
  plt.savefig(base_dir + out_file, dpi=300)
  plt.clf()

def plot_heuristic_comparison(base, bound, runtime, out_file):
  """Compare different heuristics on the same runtime and budget."""
  base_dir = util.get_output_path(ASYMPTOTICS_MOD + '/' + base, '')
  results = []
  for result_path in glob.glob(base_dir + '*.json'):
    js = None
    with open(result_path, 'r') as jf:
      js = json.loads(jf.read())
    assert js is not None
    if js['config']['runtime'] != runtime.ID: continue
    if js['config']['memory'] != str(bound): continue
    assert len(results) == 0 or results[-1]['config']['releases'] == js['config']['releases']
    assert len(results) == 0 or results[-1]['config']['ns'] == js['config']['ns']
    results.append(js)

  for res in results:
    h_str = res['config']['heuristic']
    h_color, h_marker = HEURISTICS[h_str].COLOR, HEURISTICS[h_str].MARKER
    plt.plot(res['results']['layers'], res['results']['computes'], label=h_str, \
      color=h_color, marker=h_marker, alpha=0.5, linewidth=3, ms=10)
    if res['results']['had_OOM']:
      plt.axvline(x=res['results']['layers'][0], color=h_color, linestyle='dotted', \
        alpha=0.3, linewidth=3)
    if res['results']['had_thrash'] is not None:
      plt.axvline(x=res['results']['layers'][-1], color=h_color, linestyle='--', \
        alpha=0.3, linewidth=3)
  if isinstance(bound, Log2Bound):
    plot_treeverse(results[0]['config']['ns'])
  else:
    plot_tq(results[0]['config']['ns'])
  plt.grid(True)
  plt.xlabel(r'Number of Layers $n$')
  plt.ylabel(r'Additional Compute')
  plt.legend()
  plt.title('Layers vs. Compute Overhead ({} Memory)'.format(str(bound)))
  plt.savefig(base_dir + out_file, dpi=300)
  plt.clf()
