import json, glob
import matplotlib.pyplot as plt

from simrd.heuristic import *

from ...util import *
from ...eval import models as models

from .util import *
from .definitions import *

def get_pareto_results(result_file):
  results = None
  with open(result_file, 'r') as rf:
    results = json.loads(rf.read())
  assert results is not None
  return results

def get_pareto_data(result_file, overheads=True):
  results = get_pareto_results(result_file)
  heuristic = results['config']['heuristic']
  ratios, succ_ratios, values = [], [], []
  last_ratio = 1
  fail_type = None
  multi_trial = False
  for res in results['results']:
    ratios.append(res['ratio'])
    any_OOM, any_thrash = any(res['OOM']), any(res['remat_exceeded'])
    if any_OOM or any_thrash:
      fail_type = 'OOM' if any_OOM else 'remat_exceeded'
      continue
    succ_ratios.append(res['ratio'])
    if res['num_trials'] > 1:
      multi_trial = True
    if overheads:
      rstr = 'overhead'
    else:  # accesses
      rstr = 'heuristic_access_count'
    
    mean = sum(res[rstr]) / res['num_trials']
    vmin = min(res[rstr])
    vmax = max(res[rstr])
    if res['num_trials'] > 1:
      stddev = math.sqrt(sum([(v - mean) ** 2 for v in res[rstr]]) / res['num_trials'])
    else:
      stddev = 0
    val = {'mean': mean, 'min': vmin, 'max': vmax, 'stddev': stddev}
    values.append(val)

    if res['ratio'] < last_ratio:
      last_ratio = res['ratio']

  return results, ratios, succ_ratios, values, last_ratio, fail_type, multi_trial

def plot_pareto(result_file, finalize=False, ax=None,
                title=None, desc=None, lps=None):
  if ax is not None:
    fax = ax
  else:
    fax = plt.gca()
  results, ratios, succ_ratios, overheads, last_ratio, fail_type, multi_trial \
    = get_pareto_data(result_file)
  heuristic = results['config']['heuristic']

  if lps is None:
    lps = LinePlotSettings.from_heuristic(HEURISTICS[heuristic]())

  if title is None:
    name = results['config']['model']['name']
    title = models.MANIFEST[name]['title']
    desc = models.MANIFEST[name]['desc']

  if fail_type:
    fail_linestyle = 'dotted' if fail_type == 'OOM' else 'dashed'
    fail_line = fax.axvline(x=last_ratio, alpha=0.3, linewidth=3, color=lps.color, linestyle=fail_linestyle)

  if len(succ_ratios) > 0:
    omeans = [o['mean'] for o in overheads]
    if multi_trial:
      omins = [o['min'] for o in overheads]
      omaxs = [o['max'] for o in overheads]
      fax.fill_between(succ_ratios, omins, omaxs, step='post', color=lps.color, alpha=0.3)
    fax.step(succ_ratios, omeans, alpha=0.75, linewidth=2.5, marker=None, where='post', \
      color=lps.color, linestyle=lps.linestyle)
    x, y = [succ_ratios[0], succ_ratios[-1]], [omeans[0], omeans[-1]]
    fax.scatter(x, y, zorder=3, c=lps.color, marker=lps.marker, alpha=0.75, s=9**2)

  if finalize:
    # fill in impossible regimes
    baseline_memory = results['config']['baseline_memory']
    baseline_const= results['config']['baseline_const']
    baseline_bottleneck = results['config']['baseline_bottleneck']
    const_ratio = baseline_const / baseline_memory
    bottleneck_ratio = baseline_bottleneck / baseline_memory
    fax.axvspan(0.0, const_ratio, color='black')
    fax.axvspan(const_ratio, const_ratio + bottleneck_ratio, color='grey')

    fax.axhline(1.0, label='Baseline', color='black', linestyle='--', linewidth=3, alpha=0.8)
    fax.grid(True)
    fax.text(0.98, 0.97, title, fontsize=14, weight='bold', ha='right', va='top', transform=fax.transAxes, \
      bbox=dict(facecolor='white', edgecolor='none'))
    fax.text(0.98, 0.87, desc, fontsize=12, ha='right', va='top', transform=fax.transAxes, \
      bbox=dict(facecolor='white', edgecolor='none'))
    fax.set_xticks(ratios)

def plot_accesses(result_file, finalize=False, ax=None,
                  title=None, lps=None):
  if ax is not None:
    fax = ax
  else:
    fax = plt.gca()
  results, ratios, succ_ratios, accesses, last_ratio, fail_type, multi_trial = \
    get_pareto_data(result_file, overheads=False)
  heuristic = results['config']['heuristic']

  if lps is None:
    lps = LinePlotSettings.from_heuristic(HEURISTICS[heuristic]())

  if title is None:
    title = results['config']['model']['name']
    desc = results['config']['model'].get('desc', 'placeholder text')

  if fail_type:
    fail_linestyle = 'dotted' if fail_type == 'OOM' else 'dashed'
    fail_line = fax.axvline(x=last_ratio, alpha=0.3, linewidth=3, color=lps.color, linestyle=fail_linestyle)

  accesses = [acc['min'] for acc in accesses]  # no stddev for deterministic heuristics
  fax.plot(succ_ratios, accesses, alpha=0.75, linewidth=3, ms=10, \
    color=lps.color, marker=lps.marker, linestyle=lps.linestyle, label=lps.label)

  if finalize:
    fax.grid(True)
    fax.text(0.98, 0.975, title, fontsize=14, weight='bold', ha='right', va='top', transform=fax.transAxes)
    fax.text(0.98, 0.89, desc, fontsize=12, ha='right', va='top', transform=fax.transAxes)
    fax.set_xticks(ratios)
    fax.set_yscale('log')

def plot_heuristics_f(base_dir, heuristics=None, ax=None):
  result_files = glob.glob(base_dir + '/' + '*.json')
  if heuristics is not None:
    h_strs = [type(h).__name__ for h in heuristics]
    result_files = [rf for rf in result_files if get_result_heuristic_str(rf) in h_strs]
  for i, result_path in enumerate(result_files):
    finalize = (i == len(result_files) - 1)
    plot_pareto(result_path, finalize=finalize, ax=ax)

def plot_banishing_f(base_dir, heuristics=None, ax=None):
  assert heuristics is None
  result_files = glob.glob(base_dir + '/' + '*.json')
  assert len(result_files) == 3
  for i, result_path in enumerate(result_files):
    res = get_pareto_results(result_path)
    lps = None
    if res['config']['runtime'] == 'V1':
      lps = LinePlotSettings('X', 'orange', '-', 'Banishing')
    elif res['config']['kwargs'].get('no_dealloc', False):
      lps = LinePlotSettings('+', 'red', '-', 'No Deallocations')
    else:
      lps = LinePlotSettings('*', 'green', '-', 'Eager eviction')
    plot_pareto(result_path, finalize=(i == 1), ax=ax, lps=lps)

def plot_accesses_f(base_dir, heuristics=None, ax=None):
  result_files = glob.glob(base_dir + '/' + '*.json')
  if heuristics is not None:
    h_strs = [type(h).__name__ for h in heuristics]
    result_files = [rf for rf in result_files if get_result_heuristic_str(rf) in h_strs]
  for i, result_path in enumerate(result_files):
    finalize = (i == len(result_files) - 1)
    plot_accesses(result_path, finalize=finalize, ax=ax)

def get_heuristics_handles_labels(heuristics):
  handles, labels = [], []
  for h in heuristics:
    lps = LinePlotSettings.from_heuristic(h)
    handles.append(plt.Line2D(
      [], [], linewidth=3, markersize=10, linestyle=lps.linestyle, color=lps.color, marker=lps.marker
    ))
    labels.append(lps.label)
  return handles, labels

def get_banishing_handles_labels():
  handles = [
    plt.Line2D([], [], linewidth=3, markersize=10, linestyle='-', marker='X', color='orange'),
    plt.Line2D([], [], linewidth=3, markersize=10, linestyle='-', marker='*', color='green'),
    plt.Line2D([], [], linewidth=3, markersize=10, linestyle='-', marker='+', color='red')
  ]
  labels = ['Banishing', 'Eager eviction', 'No Deallocations']
  return handles, labels

def plot_paretos_paper(base_dirs, heuristics=None, plot_f=plot_heuristics_f, legend=None):
  # there is paper-specific stuff hacked together here, beware
  plt.style.use('seaborn-paper')
  fig, ax = plt.subplots(2, 4, sharex='all', sharey='all')
  i, j = 0, 0
  for k, base_dir in enumerate(base_dirs):
    plot_f(base_dir, heuristics=heuristics, ax=ax[i, j])
    # https://stackoverflow.com/questions/20337664/cleanest-way-to-hide-every-nth-tick-label-in-matplotlib-colorbar
    [l.set_visible(False) for (i,l) in enumerate(ax[i,j].xaxis.get_ticklabels()) if i % 2 != 0]
    ax[i,j].set_ylim(bottom=0.9, top=2.0)
    ax[i,j].set_xlim(left=0.0, right=1.1)
    ax[i,j].tick_params(labelsize=14)
    j += 1
    if j > 3:
      i += 1
      j = 0
  fig.add_subplot(111, frameon=False)
  fig.set_size_inches(18, 6)
  # https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  plt.xlabel('Memory Ratio', fontsize=14, labelpad=10, fontweight='bold')
  plt.ylabel(r'Compute Overhead ($\times$)', fontsize=14, labelpad=22, fontweight='bold')
  
  # make legend
  if not legend:
    handles, labels = get_heuristics_handles_labels(heuristics)
  else:
    handles, labels = legend
  plt.legend(
    handles, labels,
    bbox_to_anchor=(0.5,0.015),
    loc='lower center',
    bbox_transform=fig.transFigure,
    ncol=7,
    borderaxespad=0,
    prop={'size': 14}
  )
  fig.tight_layout()

def plot_pareto_paper(base_dirs=None, output_dir=None, plot_file=None, num_models=len(models.MANIFEST)):
  if base_dirs is None:
    if output_dir is None:
      output_dir = get_output_dir(PARETO_MOD)
    base_dirs = get_top_k_base_dirs(num_models, output_dir)[::-1]
  if plot_file is None:
    plot_file = 'data/pareto.pdf'
  print('plotting pareto experiment...')
  print('  directories: {}'.format(base_dirs))
  plot_paretos_paper(base_dirs, heuristics=PAPER_PARETO_HEURISTICS)
  plt.savefig(plot_file, dpi=300)
  plt.clf()

def plot_ablation_paper(base_dirs=None, output_dir=None, plot_file=None, num_models=len(models.MANIFEST)):
  if base_dirs is None:
    if output_dir is None:
      output_dir = get_output_dir(PARETO_MOD)
    base_dirs = get_top_k_base_dirs(num_models, output_dir)[::-1]
  if plot_file is None:
    plot_file = 'data/ablation_{}.pdf'
  print('plotting ablation experiment...')
  print('  directories: {}'.format(base_dirs))
  heuristic_groups = [
    PAPER_ABLATION_HEURISTICS[0:4],
    PAPER_ABLATION_HEURISTICS[4:8],
    PAPER_ABLATION_HEURISTICS[8:12],
    PAPER_ABLATION_HEURISTICS[12:16]
  ]
  group_names = [
    'full_e', 'eqclass', 'local', 'none'
  ]
  for group, name in zip(heuristic_groups, group_names):
    plot_paretos_paper(base_dirs, heuristics=group)
    plt.savefig(plot_file.format(name), dpi=300)
    plt.clf()

def plot_banishing_paper(base_dirs=None, output_dir=None, plot_file=None, num_models=len(models.MANIFEST)):
  if base_dirs is None:
    if output_dir is None:
      output_dir = get_output_dir(PARETO_MOD)
    base_dirs = get_top_k_base_dirs(num_models, output_dir)[::-1]
  if plot_file is None:
    plot_file = 'data/banishing.pdf'
  print('plotting banishing experiment...')
  print('  directories: {}'.format(base_dirs))
  plot_paretos_paper(base_dirs, plot_f=plot_banishing_f, legend=get_banishing_handles_labels())
  plt.savefig(plot_file, dpi=300)
  plt.clf()

def plot_accesses_paper(base_dirs=None, output_dir=None, plot_file=None, num_models=len(models.MANIFEST)):
  if base_dirs is None:
    if output_dir is None:
      output_dir = get_output_dir(PARETO_MOD)
    base_dirs = get_top_k_base_dirs(num_models, output_dir)[::-1]
  if plot_file is None:
    plot_file = 'data/accesses.pdf'
  print('plotting accesses experiment...')
  print('  directories: {}'.format(base_dirs))
  plt.style.use('seaborn-paper')
  fig, ax = plt.subplots(2, 4, sharex='none', sharey='none')
  i, j = 0, 0
  for k, base_dir in enumerate(base_dirs):
    plot_accesses_f(base_dir, heuristics=PAPER_ACCESSES_HEURISTICS, ax=ax[i, j])
    # https://stackoverflow.com/questions/20337664/cleanest-way-to-hide-every-nth-tick-label-in-matplotlib-colorbar
    [l.set_visible(False) for (i,l) in enumerate(ax[i,j].xaxis.get_ticklabels()) if i % 2 != 0]
    ax[i,j].tick_params(labelsize=14)
    j += 1
    if j > 3:
      i += 1
      j = 0
  fig.add_subplot(111, frameon=False)
  fig.set_size_inches(18, 6)
  # https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  plt.xlabel('Memory Ratio', fontsize=14, labelpad=10, fontweight='bold')
  plt.ylabel('Storage Accesses by Heuristic', fontsize=14, labelpad=22, fontweight='bold')
  # extract the legend and put it at the bottom
  handles, labels = ax[0, 0].get_legend_handles_labels()
  plt.legend(
    handles, labels,
    bbox_to_anchor=(0.5,0.015),
    loc='lower center',
    bbox_transform=fig.transFigure,
    ncol=7,
    borderaxespad=0,
    prop={'size': 14}
  )
  fig.tight_layout()
  plt.savefig(plot_file, dpi=300)
  plt.clf()
