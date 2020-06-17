import json
import time
import glob, os
from datetime import datetime

from simrd.runtime import *
from simrd.heuristic import *
from simrd.heuristic.ablation import *
from simrd.parse import parse_file

from experiments.pareto import pareto
from experiments.util import get_output_path, ensure_output_path

import experiments.eval.models as models
from .definitions import *

def run_pareto(base, model, heuristic, ratios, runtime, overhead_limit,
               num_trials=1, verbose=True, **kwargs):
  config = {
    'model': model,
    'heuristic': str(heuristic),
    'heuristic_features': list(heuristic.FEATURES),
    'ratios': ratios,
    'overhead_limit': overhead_limit,
    'runtime': runtime.ID,
    'runtime_features': list(runtime.FEATURES),
    'kwargs': kwargs
  }

  if verbose:
    print('starting pareto trial for config: {}...'.format(json.dumps(config, indent=2)))
  else:
    # we're probably running the full eval, let's say which heuristic
    print('  - running heuristic {}...'.format(str(heuristic)), end='', flush=True)

  # get log executor callback
  log_path = model['log']
  if verbose: print('parsing log [{}]...'.format(log_path))
  callback = None
  with open(log_path, 'r') as log_f:
    callback = parse_file(log_f, start_annot=model['has_start'], pin_live=True)
  assert callback is not None
  if verbose: print('  done.')

  # run model with infinite budget to get baseline memory usage
  if verbose: print('getting baseline information...')
  rt = RuntimeV1(math.inf, Heuristic(), stats=False, trace=False)
  t = time.time()
  callback(rt)
  baseline_memory = rt.telemetry.summary['max_memory']
  baseline_compute = rt.telemetry.summary['model_compute']
  baseline_pinned = rt.telemetry.summary['model_pinned_memory']
  baseline_bottleneck = rt.telemetry.summary['bottleneck_memory']
  assert rt.telemetry.summary['remat_compute'] == 0
  if verbose:
    print('    - baseline compute:    {} ms'.format(baseline_compute / 1000000))
    print('    - baseline memory:     {} MB'.format(baseline_memory / 1000000))
    print('    - baseline pinned:     {} MB'.format(baseline_pinned / 1000000))
    print('    - baseline bottleneck: {} MB'.format(baseline_bottleneck / 1000000))
    print('  done, took {} seconds.'.format(time.time() - t))
  config['baseline_compute'] = baseline_compute
  config['baseline_memory'] = baseline_memory
  config['baseline_pinned'] = baseline_pinned
  config['baseline_bottleneck'] = baseline_bottleneck

  # run pareto, record results
  budgets = [int(baseline_memory * r) for r in ratios]
  remat_limit = baseline_compute * (overhead_limit - 1)
  assert remat_limit >= 0

  results = [{
    'ratio': ratios[i],
    'budget': budgets[i],
    'OOM': False,
    'remat_exceeded': False,
    'meta': None,
    'total_time': 0,
    'overhead': 0,
    'heuristic_eval_count': 0,
    'heuristic_access_count': 0
  } for i in range(len(ratios))]

  # average numerical values over trials, pick the last meta (or the first that fails)
  for trial in range(num_trials):
    rts = pareto(callback, budgets, heuristic, runtime, verbose=verbose, \
      remat_limit=remat_limit, trace=False, stats=False, **kwargs)
    for i, rt in enumerate(rts):
      if (results[i]['OOM'] or results[i]['remat_exceeded']):
        pass
      elif rt.OOM or rt.remat_exceeded:
        results[i]['OOM'] = rt.OOM
        results[i]['remat_exceeded'] = rt.remat_exceeded
        results[i]['meta'] = rt.meta
        results[i]['total_time'] = rt.meta['total_time']
        results[i]['overhead'] = rt.clock / baseline_compute
        results[i]['heuristic_eval_count'] = rt.telemetry.summary['heuristic_eval_count']
        results[i]['heuristic_access_count'] = rt.telemetry.summary['heuristic_access_count']
      else:
        assert rt.telemetry.summary['model_compute'] == baseline_compute
        assert rt.clock - 1 == baseline_compute + rt.telemetry.summary['remat_compute']
        results[i]['meta'] = rt.meta
        results[i]['total_time'] += rt.meta['total_time'] / num_trials
        results[i]['overhead'] += (rt.clock / baseline_compute) / num_trials
        results[i]['heuristic_eval_count'] += rt.telemetry.summary['heuristic_eval_count'] / num_trials
        results[i]['heuristic_access_count'] += rt.telemetry.summary['heuristic_access_count'] / num_trials

  date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
  base_mod = PARETO_MOD + '/' + base
  out_file = '{}-{}-{}.json'.format(date_str, model['name'], heuristic.ID)
  ensure_output_path(base_mod)
  out_path = get_output_path(base_mod, out_file)
  with open(out_path, 'w') as out_f:
    try:
      out_f.write(json.dumps({'config': config, 'results': results}, indent=2))
    except:
      import pdb; pdb.set_trace()

  if verbose: print('-> done, saved to [{}]'.format(out_path))
  else: print('done', flush=True)

  return out_path

def run_pareto_heuristics(base, model, heuristics, ratios, runtime, overhead_limit, **kwargs):
  for heuristic in heuristics:
    num_trials = heuristic.TRIALS
    run_pareto(base, model, heuristic, ratios, runtime, overhead_limit, num_trials, **kwargs)

def run_pareto_paper():
  ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  overhead_limit = 3.0
  heuristics = PAPER_PARETO_HEURISTICS
  runtime = RuntimeV2EagerOptimized

  for model_f in models.MODELS.values():
    model = model_f()
    print('running simulated pareto evaluation for {}...'.format(model['name']))
    base = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + model['name']
    t = time.time()
    run_pareto_heuristics(base, model, heuristics, ratios, runtime, overhead_limit, verbose=False)
    print('  done, took {} seconds.'.format(time.time() - t))

def run_ablation_paper():
  ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  overhead_limit = 5.0
  runtime = RuntimeV2EagerOptimized
  heuristics = PAPER_ABLATION_HEURISTICS

  for model_f in models.MODELS.values():
    model = model_f()
    print('running simulated ablation evaluation for {}...'.format(model['name']))
    base = datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + model['name'] + '-Ab'
    t = time.time()
    run_pareto_heuristics(base, model, heuristics, ratios, runtime, overhead_limit, \
      verbose=True)
    print('  done, took {} seconds.'.format(time.time() - t))

def run_banishing_paper():
  ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  overhead_limit = 5.0
  runtime = RuntimeV1

  for model_f in models.MODELS.values():
    model = model_f()
    print('running simulated banishing evaluation for {}...'.format(model['name']))
    base = datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + model['name'] + '-banish'
    t = time.time()
    run_pareto(base, model, DTRUnopt(), ratios, RuntimeV1, overhead_limit, verbose=True)
    run_pareto(base, model, DTR(), ratios, RuntimeV2EagerOptimized, overhead_limit, verbose=True)
    print('  done, took {} seconds.'.format(time.time() - t))

def run_accesses_paper():
  ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  overhead_limit = 3.0
  runtime = RuntimeV2EagerOptimized
  heuristics = PAPER_ACCESSES_HEURISTICS

  for model_f in models.MODELS.values():
    model = model_f()
    print('running simulated access overhead evaluation for {}...'.format(model['name']))
    base = datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + model['name'] + '-access'
    t = time.time()
    run_pareto_heuristics(base, model, heuristics, ratios, runtime, overhead_limit, \
      verbose=True)
    print('  done, took {} seconds.'.format(time.time() - t))
