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

MEASURED_KEYS = ['time', 'gpu_time', 'input_mem', 'model_mem', 'total_mem']

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
    if specific_params['kind'] == 'ratio':
        check_func = lambda err: lambda kv: err.get(kv[0]) == kv[1] if kv[0] != 'memory_budget' else True

    return any(map(lambda err: all(map(check_func(err), specific_params.items())), errors))


def get_report_prefix(experiment_name, specific_params, cmd_id=0):
    if experiment_name == 'dtr':
        if specific_params.get('kind') == 'ratio':
            return 'cmd-{}-dtr-ratio-{}'.format(cmd_id, specific_params['ratio'])
        else:
            return 'cmd-{}-dtr-fixed-{}-{}'.format(cmd_id, specific_params['batch_size'], specific_params['memory_budget'])
    elif experiment_name == 'baseline':
        return 'cmd-{}-baseline-{}'.format(cmd_id, specific_params['batch_size'])


def run_trials(config_dir, python_cmd,
               experiment_name, model_name,
               specific_params,
               n_inputs, n_reps,
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
    elif result['type'] == 'dtr':
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
    if method == 'dtr':
        for required_keys in ('batch_size', 'memory_budget'):
            if required_keys not in exp_config:
                return False, 'Missing {}'.format(required_keys)
        return True, ''
    elif method == 'baseline':
        return 'batch_size' in exp_config, ''


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
    baseline_config = { 'batch_size' : exp_config['batch_size'] }
    if 'extra_params' in exp_config:
        baseline_config['extra_params'] = exp_config['extra_params']
    filename = str(time.time()) + '.json'
    temp_file = prepare_out_file(os.getcwd(), filename)
    success, msg = run_trials(config_dir,
                              python_command('baseline', config),
                              'baseline', model, baseline_config,
                              config['n_inputs'], config['n_reps'],
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

        first_time = True
        conf_cnt = 0
        for combo in unfold_settings(exp_config):
            success, msg = run_trials(config_dir,
                                    python_command(combo['type'], config),
                                    combo['type'], model, combo,
                                    config['n_inputs'], config['n_reps'],
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
        if exp_config.get('kind') == 'ratio':
            exp_config['memory_budget'] = -1.0
        success, msg = validate_setting(exp_config['type'], exp_config)
        if not success:
            yield False, 'Malformate configuration for {}-{}: {}'.format(model, exp_config['type'], msg), None
        else:
            yield True, 'Success', exp_config


def compute_summary_stats(l):
    return {'mean': np.mean(l),
            'median': np.median(l),
            'std': np.std(l)}


def parse_data_file(experiment_name, model, config, specific_params, path_prefix, cmd_id=0):
    """
    Given an experiment name, model name, directory, and number of inputs,
    parses the corresponding data file if it exists and computes
    summary statistics for the (wall-clock) time, GPU time, and memory used in that data file for choice of specific settings

    Returns None and an error message if it fails
    """
    try:
        filename = '{}-{}.csv'.format(get_report_prefix(experiment_name, specific_params, cmd_id), model)
        if not check_file_exists(path_prefix, filename):
            return (None, 'Data file {} does not exist at {}'.format(filename, path_prefix))

        full_path = os.path.join(path_prefix, filename)

        report_errors = config['report_errors']

        metrics = {}

        memory_budget = None

        with open(full_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # In case of there are commands for the same model
                # that have the same values for all configurations
                idx = int(row['input'])
                measured = {
                    key: float(row[key]) for key in MEASURED_KEYS
                }

                if memory_budget is None and specific_params.get('kind') == 'ratio':
                    memory_budget = float(row['memory_budget'])
                    specific_params['memory_budget'] = memory_budget

                if idx not in metrics.keys():
                    metrics[idx] = {
                        key: [] for key in MEASURED_KEYS
                    }

                for key in MEASURED_KEYS:
                    metrics[idx][key].append(measured[key])

        summary = {
            'specific_params': specific_params
        }

        # in case everything errored out, this ensure that we will have a record of the error
        if report_errors:
            if check_error(experiment_name, model, specific_params, path_prefix):
                summary['summary'] = 'error'
                return summary, 'success'

        summary_stats = []
        for (_, stat) in metrics.items():
            summary_stats.append({
                key: compute_summary_stats(stat[key])
                for key in MEASURED_KEYS
            })

        summary['summary'] = summary_stats
        return (summary, 'success')

    except Exception as e:
        return (None, 'Encountered exception on ({}, {}): '.format(experiment_name, model) + render_exception(e))
