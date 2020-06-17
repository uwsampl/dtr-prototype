import json, glob
import matplotlib.pyplot as plt

from simrd.heuristic import *

from experiments.util import *
import experiments.eval.models as models

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
  fail_type, fail_line = None, None
  for res in results['results']:
    ratios.append(res['ratio'])
    if res['OOM'] or res['remat_exceeded']:
      fail_type = 'OOM' if res['OOM'] else 'remat_exceeded'
      continue
    succ_ratios.append(res['ratio'])
    if overheads:
      values.append(res['overhead'])
    else:
      # accesses
      values.append(res['heuristic_access_count'])
    if res['ratio'] < last_ratio:
      last_ratio = res['ratio']

  return results, ratios, succ_ratios, values, last_ratio, fail_type

def plot_pareto(result_file, finalize=False, ax=None,
                title=None, lps=None):
  if ax is not None:
    fax = ax
  else:
    fax = plt.gca()
  results, ratios, succ_ratios, overheads, last_ratio, fail_type = get_pareto_data(result_file)
  heuristic = results['config']['heuristic']

  if lps is None:
    lps = LinePlotSettings.from_heuristic(HEURISTICS[heuristic]())

  if title is None:
    title = '{} ({} batch, {} layers)'.format(
      results['config']['model']['name'],
      results['config']['model']['batch_size'],
      results['config']['model']['layers']
    )

  if fail_type:
    fail_linestyle = 'dotted' if fail_type == 'OOM' else 'dashed'
    fail_line = fax.axvline(x=last_ratio, alpha=0.3, linewidth=3, color=lps.color, linestyle=fail_linestyle)

  fax.plot(succ_ratios, overheads, alpha=0.75, linewidth=3, ms=10, \
    color=lps.color, marker=lps.marker, linestyle=lps.linestyle, label=lps.label)

  if finalize:
    # fill in impossible regimes
    baseline_memory = results['config']['baseline_memory']
    baseline_pinned = results['config']['baseline_pinned']
    baseline_bottleneck = results['config']['baseline_bottleneck']
    pin_ratio = baseline_pinned / baseline_memory
    bottleneck_ratio = baseline_bottleneck / baseline_memory
    fax.axvspan(0.0, pin_ratio, color='black')
    fax.axvspan(pin_ratio, pin_ratio + bottleneck_ratio, color='grey')

    fax.axhline(1.0, label='Baseline', color='black', linestyle='--', linewidth=3, alpha=0.8)
    fax.grid(True)
    fax.set_title(title, fontsize=11)
    fax.set_xticks(ratios)

def plot_accesses(result_file, finalize=False, ax=None,
                  title=None, lps=None):
  if ax is not None:
    fax = ax
  else:
    fax = plt.gca()
  results, ratios, succ_ratios, accesses, last_ratio, fail_type = \
    get_pareto_data(result_file, overheads=False)
  heuristic = results['config']['heuristic']

  if lps is None:
    lps = LinePlotSettings.from_heuristic(HEURISTICS[heuristic]())

  if title is None:
    title = '{} ({} batch, {} layers)'.format(
      results['config']['model']['name'],
      results['config']['model']['batch_size'],
      results['config']['model']['layers']
    )

  if fail_type:
    fail_linestyle = 'dotted' if fail_type == 'OOM' else 'dashed'
    fail_line = fax.axvline(x=last_ratio, alpha=0.3, linewidth=3, color=lps.color, linestyle=fail_linestyle)

  fax.plot(succ_ratios, accesses, alpha=0.75, linewidth=3, ms=10, \
    color=lps.color, marker=lps.marker, linestyle=lps.linestyle, label=lps.label)

  if finalize:
    fax.grid(True)
    fax.set_title(title, fontsize=11)
    fax.set_xticks(ratios)
    fax.set_yscale('log')

def plot_heuristics_f(base, heuristics=None, ax=None):
  base_mod = PARETO_MOD + '/' + base
  result_files = glob.glob(get_output_path(base_mod, '*.json'))
  if heuristics is not None:
    h_strs = [str(h) for h in heuristics]
    result_files = [rf for rf in result_files if get_result_heuristic_str(rf) in h_strs]
  for i, result_path in enumerate(result_files):
    finalize = (i == len(result_files) - 1)
    plot_pareto(result_path, finalize=finalize, ax=ax)

def plot_banishing_f(base, heuristics=None, ax=None):
  assert heuristics is None
  base_mod = PARETO_MOD + '/' + base
  result_files = glob.glob(get_output_path(base_mod, '*.json'))
  assert len(result_files) == 2
  for i, result_path in enumerate(result_files):
    res = get_pareto_results(result_path)
    lps = None
    if res['config']['runtime'] == 'V1':
      lps = LinePlotSettings('X', 'orange', '-', 'Banishing')
    else:
      lps = LinePlotSettings('*', 'green', '-', 'Eager eviction')
    plot_pareto(result_path, finalize=(i == 1), ax=ax, lps=lps)

def plot_accesses_f(base, heuristics=None, ax=None):
  base_mod = PARETO_MOD + '/' + base
  result_files = glob.glob(get_output_path(base_mod, '*.json'))
  if heuristics is not None:
    h_strs = [str(h) for h in heuristics]
    result_files = [rf for rf in result_files if get_result_heuristic_str(rf) in h_strs]
  for i, result_path in enumerate(result_files):
    finalize = (i == len(result_files) - 1)
    plot_accesses(result_path, finalize=finalize, ax=ax)

def plot_paretos_paper(bases, heuristics=None, plot_f=plot_heuristics_f):
  # there is paper-specific stuff hacked together here, beware
  plt.style.use('seaborn-paper')
  fig, ax = plt.subplots(2, 3, sharex='all', sharey='all')
  i, j = 0, 0
  for k, base in enumerate(bases):
    plot_f(base, heuristics=heuristics, ax=ax[i, j])
    # https://stackoverflow.com/questions/20337664/cleanest-way-to-hide-every-nth-tick-label-in-matplotlib-colorbar
    [l.set_visible(False) for (i,l) in enumerate(ax[i,j].xaxis.get_ticklabels()) if i % 2 != 0]
    ax[i,j].set_ylim(bottom=0.9, top=2.0)
    ax[i,j].set_xlim(left=0.0, right=1.1)
    ax[i,j].tick_params(labelsize=11)
    j += 1
    if j > 2:
      i += 1
      j = 0
  fig.add_subplot(111, frameon=False)
  fig.set_size_inches(12, 6)
  # https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  plt.xlabel('Memory Ratio', fontsize=14, labelpad=10)
  plt.ylabel(r'Compute Overhead ($\times$)', fontsize=14, labelpad=10)
  # extract the legend and put it at the bottom
  handles, labels = ax[0, 0].get_legend_handles_labels()
  plt.legend(
    handles, labels,
    bbox_to_anchor=(0.5,0.02),
    loc='lower center',
    bbox_transform=fig.transFigure,
    ncol=7,
    borderaxespad=0,
    prop={'size': 10}
  )
  fig.tight_layout()

def plot_pareto_paper(bases=None):
  if bases is None:
    bases = get_top_k_bases(len(models.MODELS))[::-1]
  plot_paretos_paper(bases, heuristics=PAPER_PARETO_HEURISTICS)
  plt.savefig('data/pareto.pdf')
  plt.clf()

def plot_ablation_paper(bases=None):
  if bases is None:
    bases = get_top_k_bases(len(models.MODELS))[::-1]
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
    plot_paretos_paper(bases, heuristics=group)
    plt.savefig('data/ablation_{}.pdf'.format(name))
    plt.clf()

def plot_banishing_paper(bases=None):
  if bases is None:
    bases = get_top_k_bases(len(models.MODELS))[::-1]
  plot_paretos_paper(bases, plot_f=plot_banishing_f)
  plt.savefig('data/banishing.pdf')
  plt.clf()

def plot_accesses_paper(bases=None):
  if bases is None:
    bases = get_top_k_bases(len(models.MODELS))[::-1]
  plt.style.use('seaborn-paper')
  fig, ax = plt.subplots(2, 3, sharex='none', sharey='none')
  i, j = 0, 0
  for k, base in enumerate(bases):
    plot_accesses_f(base, heuristics=PAPER_ACCESSES_HEURISTICS, ax=ax[i, j])
    # https://stackoverflow.com/questions/20337664/cleanest-way-to-hide-every-nth-tick-label-in-matplotlib-colorbar
    [l.set_visible(False) for (i,l) in enumerate(ax[i,j].xaxis.get_ticklabels()) if i % 2 != 0]
    ax[i,j].tick_params(labelsize=11)
    j += 1
    if j > 2:
      i += 1
      j = 0
  fig.add_subplot(111, frameon=False)
  fig.set_size_inches(12, 6)
  # https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
  plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
  plt.xlabel('Memory Ratio', fontsize=14, labelpad=10)
  plt.ylabel('Storage Accesses by Heuristic', fontsize=14, labelpad=10)
  # extract the legend and put it at the bottom
  handles, labels = ax[0, 0].get_legend_handles_labels()
  plt.legend(
    handles, labels,
    bbox_to_anchor=(0.5,0.02),
    loc='lower center',
    bbox_transform=fig.transFigure,
    ncol=7,
    borderaxespad=0,
    prop={'size': 10}
  )
  fig.tight_layout()
  plt.savefig('data/accesses.pdf')
  plt.clf()
