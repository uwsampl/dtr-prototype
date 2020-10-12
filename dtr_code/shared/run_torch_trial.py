"""
To avoid any issues of memory hanging around between inputs,
we run each input as a separate process.

A little ugly but effective
"""
import glob
import os
import random
import time

import torch
import gc

from common import invoke_main, read_json, write_json, prepare_out_file, check_file_exists

from validate_config import validate_trials_config
from pt_trial_util import create_csv_writer
from tqdm import tqdm
import model_util
import json


def extend_simrd_config(dest_dir, sim_conf_filename, model_name, specific_params, log_name):
    if not check_file_exists(dest_dir, sim_conf_filename):
        prepare_out_file(dest_dir, sim_conf_filename)
        write_json(dest_dir, sim_conf_filename, dict())

    conf = read_json(dest_dir, sim_conf_filename)
    if model_name not in conf:
        conf[model_name] = []
    conf[model_name].append({
            'name': model_util.get_model_family(model_name),
            'batch_size': str(specific_params['batch_size']),
            'layers': specific_params.get('layers', model_util.get_model_layers(model_name)),
            'type': model_util.get_model_type(model_name),
            'log': log_name,
            'has_start': True
    })
    write_json(dest_dir, sim_conf_filename, conf)


def save_trial_log(dest_dir, sim_conf_filename, model_name, specific_params, is_baseline=False):
    """
    Find the last DTR log produced in the trial (if any exist)
    and move it to the directory
    """
    all_logs = glob.glob(os.path.join(os.getcwd(), '*.log'))
    if not all_logs:
        return

    # if we delete all logs in advance, there should be at most one log
    assert len(all_logs) == 1
    most_recent = all_logs[0]

    # rename and move
    # (new name just appends info to the old one)
    batch_size = specific_params['batch_size']
    budget = specific_params['memory_budget']
    if budget < 0:
        budget = 'inf'
    new_name = '{}-{}-{}-{}'.format(model_name, batch_size, budget,
                                    os.path.basename(most_recent))
    filename = prepare_out_file(dest_dir, new_name)
    os.rename(most_recent, filename)
    if is_baseline and sim_conf_filename is not None:
        extend_simrd_config(dest_dir, sim_conf_filename, model_name, specific_params, filename)


def delete_logs():
    for log in glob.glob(os.path.join(os.getcwd(), '*.log')):
        os.remove(log)


def run_single_measurement(model_name, produce_model, run_model, teardown, inp, criterion, extra_params, use_dtr, use_profiling):
    """
    This function initializes a model and performs
    a single measurement of the model on the given input.

    While it might seem most reasonable to initialize
    the model outside of the loop, DTR's logs have shown
    that certain constants in the model persist between loop iterations;
    performing these actions in a separate *function scope* turned out to be the only
    way to prevent having those constants hang around.

    Returns a dict of measurements
    """
    torch.cuda.reset_max_memory_allocated()
    # resetting means the count should be reset to
    # only what's in scope, meaning only the input
    input_mem = torch.cuda.max_memory_allocated()
    model = produce_model(extra_params=extra_params)
    params = []
    for m in model:
        if hasattr(m, 'parameters'):
            params.extend(m.parameters())

    model_mem = torch.cuda.max_memory_allocated()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # start timing
    torch.cuda.synchronize()
    start_time = time.time()
    if use_dtr:
        torch.reset_profile()
    start.record()
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    run_model(criterion, *model, *inp)
    end.record()
    start_sync = time.time()
    torch.cuda.synchronize()
    end_sync = time.time()
    end_time = time.time()
    # end timing

    if use_dtr:
        # operators-only time, tracked by DTR
        cuda_time = torch.compute_time()

    base_compute_time = -1
    remat_compute_time = -1
    search_time = -1
    cost_time = -1
    if use_profiling:
        base_compute_time = torch.base_compute_time()
        remat_compute_time = torch.remat_compute_time()
        search_time = torch.search_time()
        cost_time = torch.cost_time()
        torch.reset_profile()

    total_mem = torch.cuda.max_memory_allocated()
    teardown(*model)
    torch.cuda.reset_max_memory_allocated()

    del model

    if use_dtr:
        torch.toggle_log(False)

    del params

    result = {
        'time': end_time - start_time,
        'sync_time': end_sync - start_sync,
        'gpu_time': start.elapsed_time(end),
        'input_mem': input_mem,
        'model_mem': model_mem,
        'total_mem': total_mem,
        'base_compute_time': base_compute_time,
        'remat_compute_time': remat_compute_time,
        'search_time': search_time,
        'cost_time': cost_time
    }
    if use_dtr:
        result['cuda_time'] = cuda_time
    else:
        result['cuda_time'] = -1.0


    return result


def timing_loop(model_name, i, config, use_dtr,
                specific_params, writer, trial_run=False, trial_run_outfile=None, memory_budget=-1.0):
    dry_run = config['dry_run']
    measurements = []
    print(f'Running {model_name} : {specific_params}')

    # remove any logs hanging around (so we only have to look for one)
    delete_logs()

    # we only save logs for the final input on DTR
    save_log = use_dtr and specific_params.get('save_logs', config['save_logs']) and i == config['n_inputs'] - 1
    if use_dtr:
        torch.toggle_log(False)

    # whether to report profiling info
    use_profiling = use_dtr and specific_params.get('use_profiling', False)

    use_cudnn = model_util.use_cudnn(model_name)
    with torch.backends.cudnn.flags(enabled=use_cudnn, benchmark=use_cudnn):
        criterion = model_util.get_criterion(model_name)
        produce_model, gen_input, run_model, teardown = model_util.prepare_model(model_name,
                                                                                 specific_params['batch_size'],
                                                                                 use_dtr=use_dtr)
        inp = gen_input(i, specific_params.get('extra_params', dict()))

        n_reps = specific_params.get('n_reps', config['n_reps'])

        if use_profiling:
            torch.toggle_profile(use_profiling)

        progress = tqdm(range(dry_run + n_reps))
        for j in progress:
            progress.set_description(f'Rep [{j}]' + '' if j > dry_run else f'Dry run [{j}]')
            gc.collect()
            # Annotate where the final run starts in the log
            if save_log and j == dry_run + n_reps - 1:
                torch.toggle_log(True)
                torch.annotate_log('START')

            res = run_single_measurement(model_name, produce_model, run_model,
                                         teardown, inp, criterion, extra_params=specific_params.get('extra_params', dict()), use_dtr=use_dtr, use_profiling=use_profiling)
            if j >= dry_run:
                measurements.append(res)

    # write to csv file only when this trial is not
    # for getting a baseline memory usage
    if trial_run:
        write_json(os.getcwd(), trial_run_outfile, {
            'mem' : max(map(lambda data: data['total_mem'], measurements))
        })
        return

    if save_log:
        save_trial_log(config['log_dest'], config.get('simrd_config', None),
                       model_name,
                       specific_params,
                       is_baseline=specific_params['memory_budget'] == -1)

    # clean up after ourselves
    delete_logs()

    # do all the writing after the trial is over
    for j in range(len(measurements)):
        data = measurements[j]
        # do unit conversions now: times in ms,
        # memory in MB
        writer.writerow({
            'time': data['time']*1e3,
            'sync_time': data['sync_time']*1e3,
            # pytorch's cuda elapsed time is already in ms
            'gpu_time': float(data['gpu_time']),
            # 'cuda_time' : float(data['cuda_time']) * 1e-6,
            'input_mem': data['input_mem']*1e-6,
            'model_mem': data['model_mem']*1e-6,
            'total_mem': data['total_mem']*1e-6,
            'memory_budget': memory_budget,
            # profiling (reported in nanoseconds)
            'base_compute_time': data['base_compute_time']*1e-6,
            'remat_compute_time': data['remat_compute_time']*1e-6,
            'search_time': data['search_time']*1e-6,
            'cost_time': data['cost_time']*1e-6,
            'rep': j - dry_run,
            'input': i,
            **specific_params
         })


def main(config_dir, experiment_mode, model_name, input_idx, params_file, out_file,
         trial_run=False, trial_run_outfile=None):
    config, msg = validate_trials_config(config_dir)
    if config is None:
        print(msg)
        return 1

    use_dtr = (experiment_mode == 'dtr')

    i = int(input_idx)
    is_trial = trial_run == 'True'

    if config['set_seed']:
        torch.manual_seed(config['seed'] + i)
        random.seed(config['seed'] + i)

    cwd = os.getcwd()

    # handle specific params, esp. for DTR
    specific_params = read_json(cwd, params_file)
    assert 'batch_size' in specific_params
    if use_dtr:
        assert 'memory_budget' in specific_params
        if specific_params['memory_budget'] > 0:
            print(f'Setting budget to {int(specific_params["memory_budget"])}')
            torch.set_memory_budget(int(specific_params['memory_budget']))
    if is_trial:
        timing_loop(model_name, i, config, use_dtr, specific_params, None, True, trial_run_outfile)
        return

    with open(out_file, 'a', newline='') as csvfile:
        writer = create_csv_writer(csvfile, specific_params)
        timing_loop(model_name, i, config, use_dtr, specific_params, writer, memory_budget=specific_params.get('memory_budget', -1))


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'experiment_mode',
                'model_name', 'input_idx', 'params_file',
                'out_file', 'trial_run', 'trial_run_outfile')
