"""
Utilities for setting up PyTorch memory usage experiments.
"""
import csv
from itertools import product as iter_product
import os
import subprocess
import time

import numpy as np

from common import (check_file_exists, prepare_out_file,
                    read_json, render_exception, write_json)

MEASURED_KEYS = ['time', 'gpu_time', 'input_mem', 'model_mem', 'total_mem', 'sync_time',
                 # profiling output
                 'base_compute_time', 'remat_compute_time', 'search_time', 'cost_time', 'memory_budget']

def create_csv_writer(csvfile, specific_params):
    fieldnames = ['input', 'rep'] + MEASURED_KEYS + list(specific_params.keys())
    return csv.DictWriter(csvfile, fieldnames=fieldnames)


def python_command(setting, config):
    if setting == 'dtr':
        return os.path.expanduser(config['dtr_torch_cmd'])
    return 'python3'


def log_error(experiment_name, model_name, specific_params, inp, err_msg, path_prefix):
    err_info = {
        'input': inp,
        'msg': err_msg
    }

    logged_errors = {}
    if check_file_exists(path_prefix, 'errors.json'):
        logged_errors = read_json(path_prefix, 'errors.json')
    if experiment_name not in logged_errors:
        logged_errors[experiment_name] = {}
    if model_name not in logged_errors[experiment_name]:
        logged_errors[experiment_name][model_name] = []
    logged_errors[experiment_name][model_name].append({
        'err_info': err_info,
        **specific_params
    })
    write_json(path_prefix, 'errors.json', logged_errors)


def check_error(experiment_name, model_name, specific_params, path_prefix):
    if not check_file_exists(path_prefix, 'errors.json'):
        return False
    logged_errors = read_json(path_prefix, 'errors.json')
    if experiment_name not in logged_errors:
        return False
    if model_name not in logged_errors[experiment_name]:
        return False
    errors = logged_errors[experiment_name][model_name]

    check_func = lambda err: lambda kv: err.get(kv[0]) == kv[1]
    if specific_params.get('kind') == 'ratio':
        check_func = lambda err: lambda kv: err.get(kv[0]) == kv[1] if kv[0] != 'memory_budget' else True

    return any(map(lambda err: all(map(check_func(err), specific_params.items())), errors))


def get_report_prefix(experiment_name, specific_params, cmd_id=0):
    if experiment_name == 'dtr':
        if specific_params.get('kind') == 'ratio':
            return 'cmd-{}-dtr-ratio-{}'.format(cmd_id, specific_params['ratio'])
        elif specific_params.get('kind') == 'fixed':
            return 'cmd-{}-dtr-fixed-{}-{}'.format(cmd_id, specific_params['batch_size'], specific_params['memory_budget'])
        elif specific_params.get('kind') == 'param_sweep':
            return 'cmd-{}-dtr-sweep-{}-{}'.format(cmd_id, specific_params['batch_size'], specific_params['memory_budget'])
    elif experiment_name == 'baseline':
        return 'cmd-{}-baseline-{}'.format(cmd_id, specific_params['batch_size'])


def run_trials(config_dir, python_cmd,
               experiment_name, model_name,
               specific_params,
               n_inputs,
               path_prefix,
               report_errors=False,
               append_to_csv=False,
               trial_run=False,
               trial_run_outfile='',
               cmd_id=0,
               conf_cnt=0):
    """
    Responsible for recording the time and max memory usage
    from running a model (the user must provide a lambda for
    actually running the model because different kinds of models
    need different kinds of setup and a lambda that generates an
    input for running that model)

    :params:
        trial_run: When set to true, no persistent experiment data will be saved. It is used to
                   run a baseline trial and record how much memory is used then set the memory budget
                   for `ratio` commands of DTR experiments

        trial_run_out_file: the temporary file that stores the memory usage data of the baseline run

        cmd_id: the command id for current model, starting from 0 by default
        conf_cnt: the id of confguration generated from `unfold_settings`; this is used for tracking
                  which exact configuration that caused errors. 
    """
    try:
        cwd = os.getcwd()
        params_file = 'specific_params.json'
        try:
            write_json(cwd, params_file, specific_params)
            if not trial_run:
                filename = prepare_out_file(path_prefix,
                                            '{}-{}.csv'.format(get_report_prefix(experiment_name, specific_params, cmd_id), model_name))
                mode = 'a' if append_to_csv else 'w'
                with open(filename, mode, newline='') as csvfile:
                    writer = create_csv_writer(csvfile, specific_params)
                    if not append_to_csv:
                        writer.writeheader()
            else:
                filename = ''

            shared_dir = os.path.dirname(os.path.abspath(__file__))
            run_script = os.path.join(shared_dir, 'run_torch_trial.py')

            for i in range(n_inputs):
                try:
                    subprocess.run(
                        [python_cmd, run_script,
                         '--config-dir', config_dir,
                         '--experiment-mode', experiment_name,
                         '--model-name', model_name,
                         '--input-idx', str(i),
                         '--params-file', params_file,
                         '--out-file', filename,
                         '--trial-run', str(trial_run),
                         '--trial-run-outfile', trial_run_outfile
                        ],
                        check=True, timeout=specific_params.get('timeout', 60))
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    if not report_errors:
                        raise e
                    if trial_run:
                        return (False, 'Baseline failed: {}'.format(render_exception(e)))
                    log_error(experiment_name, model_name, specific_params, i, render_exception(e), path_prefix)
                    return (True, 'successfully caught error')
                time.sleep(4)
            return (True, 'success')
        finally:
            os.remove(params_file)
    except Exception as e:
        return (False,
                'Encountered exception on ({}, {}, {}):\n'.format(
                    experiment_name, model_name, specific_params) + render_exception(e))


def process_command(command: dict, config_template: dict):
    '''
        Generate a setting with all necessary fields according
        to the default setting and the type of the commands.

        :params:
            command: the command provided in the config.json
            config_template: default values for the settings that is required
                             while running an experiment
    '''
    result = command.copy()

    if result['type'] == 'baseline':
        if 'batch_size' not in result:
            result['batch_size'] = config_template['batch_size']
        return result
    elif result['type'] == 'dtr' or result['type'] == 'simrd':
        for (k, v) in config_template.items():
            if k not in command:
                result[k] = v
        return result
    else:
        raise Exception('Unknown type: {}'.format(result['type']))


def validate_setting(method, exp_config):
    '''
        Check whether the settings for an experiment contains
        all the required values
    '''
    def check_dtr():
        for required_keys in ('batch_size', 'memory_budget'):
            if required_keys not in exp_config:
                return False, 'Missing {}'.format(required_keys)
        return True, ''

    def check_simrd():
        if exp_config['kind'] == 'use_log':
            return 'file_path' in exp_config and 'config' in exp_config, ''
        elif exp_config['kind'] == 'get_log':
            return check_dtr()
        else:
            raise Exception(f'unknown kind: {exp_config["kind"]}')

    return {
        'dtr' : check_dtr,
        'simrd': check_simrd,
        'baseline': lambda: ('batch_size' in exp_config, '')
    }.get(method, lambda: (False, 'unknown kind'))()


def unfold_settings(exp_config):
    '''
        Unfold a command and get all possible
        settings. Returned as an Iterable.
        The possible settings are generated by taking
        a Cartesian product over list fields of the command

        Note: for `ratio` command, the `memory_budget` is calculated here in order to
              avoid multiple runs of baseline trial
    '''
    setting_heading = list()
    list_fields = list()
    for (k, v) in exp_config.items():
        if isinstance(v, list):
            setting_heading.append(k)
            list_fields.append(v)

    if not list_fields:
        yield exp_config
    else:
        for combo in iter_product(*list_fields):
            # necessary to copy each time
            # since the old data might be used later
            result = exp_config.copy()
            for i in range(len(list_fields)):
                result[setting_heading[i]] = combo[i]
            if result.get('kind') == 'ratio':
                result['memory_budget'] *= result['ratio']
            yield result


def run_baseline(model, exp_config, config, config_dir, output_dir):
    '''
        Run a baseline triral and obtain memory usage.
        This is used for getting a reference memory usage for
        DTR `ratio` commands
    '''
    baseline_config = { 'batch_size' : exp_config['batch_size'],
                        'timeout': exp_config.get('timeout', 60),
                        # only doing a minimal number of runs because we are only getting the memory usage,
                        # which should be identical between runs
                        'n_reps': 10,
                        'extra_params': exp_config.get('extra_params', {})
    }
    if 'input_params' in exp_config:
        baseline_config['input_params'] = exp_config['input_params']
    filename = str(time.time()) + '.json'
    temp_file = prepare_out_file(os.getcwd(), filename)
    success, msg = run_trials(config_dir,
                              python_command('baseline', config),
                              'baseline', model, baseline_config,
                              exp_config.get('n_inputs', config['n_inputs']),
                              output_dir,
                              report_errors=config['report_errors'],
                              append_to_csv=False,
                              trial_run=True,
                              trial_run_outfile=temp_file)
    if not success:
        return False, 'Error while running baseline trial: \n{}'.format(msg)

    mem_usage = read_json(output_dir, temp_file)
    os.remove(temp_file)
    if 'mem' not in mem_usage:
        return False, 'failed to get baseline memory usage'
    return True, mem_usage['mem']

def eval_command(model, exp_config, config, config_dir, output_dir, cmd_id):
    try:
        if exp_config.get('kind') == 'ratio':
            success, result = run_baseline(model, exp_config, config, config_dir, output_dir)
            if not success:
                return False, result

            # the actual memory budget calculation is
            # in `unfold_settings`
            exp_config['memory_budget'] = result

        conf_cnt = 0
        for combo in unfold_settings(exp_config):
            success, msg = run_trials(config_dir,
                                    python_command(combo['type'], config),
                                    combo['type'], model, combo,
                                    config['n_inputs'],
                                    output_dir,
                                    report_errors=config['report_errors'],
                                    append_to_csv=False,
                                    trial_run=False,
                                    cmd_id=cmd_id,
                                    conf_cnt=conf_cnt)
            if not success:
                return False, msg
            conf_cnt += 1
        return True, 'success'
    except Exception as e:
        return (False,
                'Encountered outer iteration exception:\n' + render_exception(e))


def parse_commands(model, config):
    '''
        Parse a command and return a processed command, which
        can be used to generate settings for experiments

        :params:
            model: the name of the model
            config: the top-level config
    '''
    if not config['dtr_settings'].get(model):
        yield False, 'No settings for {}'.format(model), None

    default_setting = config['dtr_settings'].get('default')
    model_commands = config['dtr_settings'].get(model)

    if default_setting is not None:
        config_template = default_setting.copy()
    else:
        config_template = dict()

    for command in model_commands:
        exp_config = process_command(command, config_template)
        if exp_config.get('kind') in ('ratio',):
            exp_config['memory_budget'] = -1.0
        success, msg = validate_setting(exp_config['type'], exp_config)
        if not success:
            yield False, 'Malformat configuration for {}-{}: {}'.format(model, exp_config['type'], msg), None
        else:
            yield True, 'Success', exp_config


def bootstrap_conf_intervals(data, stat, bootstrap_iters=10000, confidence=95, measure='mean'):
    """
    Given an array of floats, performs bootstrap resampling for the specified number
    of iterations to estimate confidence intervals.
    """
    summary_stat = None
    if measure == 'mean':
        summary_stat = np.mean
    elif measure == 'median':
        summary_stat = np.median
    else:
        raise Exception(f'Invalid measure, must be mean or median but received {measure}')
    assert summary_stat is not None

    estimates = [
        summary_stat(np.random.choice(data, replace=True, size=len(data)))
        for i in range(bootstrap_iters)
    ]

    # To get C% confidence intervals, we exclude the bottom (100-C)/2 % and the top (100-C)/2 %
    conf_span = (100 - confidence) / 2
    return (np.percentile(estimates, conf_span), np.percentile(estimates, 100 - conf_span))


def compute_summary_stats(l, bootstrap=False):
    summary = {
        'mean': np.mean(l),
        'median': np.median(l),
        'std': np.std(l)
    }
    if bootstrap:
        summary['mean_conf'] = bootstrap_conf_intervals(l, summary['mean'], measure='mean')
        summary['median_conf'] = bootstrap_conf_intervals(l, summary['median'], measure='median')
    return summary


def collect_raw_measurements(experiment_name, model, specific_params, path_prefix, cmd_id):
    """
    Reads the raw data for the given experiment name and params and returns a tuple (metrics dictionary, memory budget if applicable, error message if there is no data file).

    The first two fields will be None if there is no data file.
    """
    filename = '{}-{}.csv'.format(get_report_prefix(experiment_name, specific_params, cmd_id), model)
    if not check_file_exists(path_prefix, filename):
        return (None, None, 'Data file {} does not exist at {}'.format(filename, path_prefix))

    full_path = os.path.join(path_prefix, filename)

    metrics = {}

    memory_budget = None

    with open(full_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # In case there are commands for the same model
            # that have the same values for all configurations
            idx = int(row['input'])
            measured = {
                key: float(row[key]) for key in MEASURED_KEYS
            }

            if memory_budget is None and specific_params.get('kind') == 'ratio':
                memory_budget = float(row['memory_budget'])

            if idx not in metrics.keys():
                metrics[idx] = {
                    key: [] for key in MEASURED_KEYS
                }

            for key in MEASURED_KEYS:
                metrics[idx][key].append(measured[key])

    return (metrics, memory_budget, 'success')


def compute_slowdowns(exp_times, baseline_times):
    """
    Given arrays of prototype times and baseline times of the same length,
    returns an array of slowdowns
    """
    return [exp_times[i]/baseline_times[i] for i in range(len(exp_times))]

def compute_throughputs(batch_size, gpu_times):
    """
    Given a batch size and an array of time running on GPU,
    returns an array of throughputs
    """
    return [batch_size / gpu_times[i] * 1000 for i in range(len(gpu_times))]

def parse_data_file(experiment_name, model, config, specific_params, path_prefix, cmd_id=0, baseline_params=None):
    """
    Given an experiment name, model name, directory, and number of inputs,
    parses the corresponding data file if it exists and computes
    summary statistics for the (wall-clock) time, GPU time, and memory used in that data file for choice of specific settings

    baseline_params: If the command is a ratio command, this will use
    the baseline to compute the slowdown per data point
    in order to better measure its distribution.

    Returns None and an error message if it fails
    """
    try:
        report_errors = config['report_errors']
        metrics, budget, msg = collect_raw_measurements(experiment_name, model, specific_params, path_prefix, cmd_id)
        if metrics is None:
            return (None, msg)

        if budget is not None and specific_params.get('kind') == 'ratio':
            specific_params['memory_budget'] = float(budget)

        summary = {
            'specific_params': specific_params
        }

        # in case everything errored out, this ensure that we will have a record of the error
        if report_errors:
            if check_error(experiment_name, model, specific_params, path_prefix):
                summary['summary'] = 'error'
                return summary, 'success'

        # if this was a ratio experiment
        # and we have a baseline available, let's compute
        # the slowdown per data point, head to head
        # and bootstrap confidence intervals
        if (specific_params.get('type') != 'baseline'
            and specific_params.get('kind') == 'ratio'
            and baseline_params is not None):

            baseline_metrics, _, baseline_msg = collect_raw_measurements(baseline_params['type'], model, baseline_params['specific_params'], path_prefix, baseline_params['cmd_id'])
            if baseline_metrics is None:
                return (None, baseline_msg)

            # compute slowdown in metrics
            for i in range(config['n_inputs']):
                dtr_times = metrics[i]['gpu_time']
                baseline_times = baseline_metrics[i]['gpu_time']
                assert len(dtr_times) == len(baseline_times)
                metrics[i]['slowdown'] = compute_slowdowns(dtr_times, baseline_times)

        # Compute throughputs for baseline param_sweep commands
        if specific_params.get('kind') == 'param_sweep' or specific_params.get('type') == 'baseline':
            for i in range(config['n_inputs']):
                metrics[i]['throughput'] = compute_throughputs(specific_params['batch_size'], metrics[i]['gpu_time'])

        summary_stats = []
        for (_, stat) in metrics.items():
            summary_dict = {
                key: compute_summary_stats(stat[key], bootstrap=('time' in key))
                for key in MEASURED_KEYS
            }
            if 'slowdown' in stat:
                summary_dict['slowdown'] = compute_summary_stats(stat['slowdown'], bootstrap=True)

            if 'throughput' in stat:
                summary_dict['throughput'] = compute_summary_stats(stat['throughput'], bootstrap=True)
               
            summary_stats.append(summary_dict)

        summary['summary'] = summary_stats
        return (summary, 'success')

    except Exception as e:
        return (None, 'Encountered exception on ({}, {}): '.format(experiment_name, model) + render_exception(e))
