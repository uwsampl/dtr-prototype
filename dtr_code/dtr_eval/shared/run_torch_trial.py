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

from common import invoke_main, read_json, write_json, prepare_out_file

from validate_config import validate
from pt_trial_util import create_csv_writer
from tqdm import tqdm
import model_util
import sys

sys.setrecursionlimit(114514)


def save_trial_log(dest_dir, model_name, specific_params):
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
    os.rename(most_recent, prepare_out_file(dest_dir, new_name))


def delete_logs():
    for log in glob.glob(os.path.join(os.getcwd(), '*.log')):
        os.remove(log)


def run_single_measurement(model_name, produce_model, run_model, teardown, inp, criterion, extra_params, use_dtr):
    """
    This function initializes a model and performs
    a single measurement of the model on the given input.

    While it might seem most reasonable to initialize
    the model outside of the loop, DTR's logs have shown
    that certain constants in the model persist between loop iterations;
    performing these actions in a separate *function scope* turned out to be the only
    way to prevent having those constants hang around.

    Returns a tuple (CPU time, GPU time, peak memory usage)
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
        torch.reset_compute_time()
    start.record()
    run_model(criterion, *model, *inp)
    end.record()
    torch.cuda.synchronize()
    end_time = time.time()
    # end timing

    if use_dtr:
        # operators-only time, tracked by DTR
        cuda_time = torch.compute_time()

    total_mem = torch.cuda.max_memory_allocated()
    teardown(*model)
    torch.cuda.reset_max_memory_allocated()

    del model

    if use_dtr:
        torch.toggle_log(False)
    
    del params

    result = {'time': end_time - start_time,
              'gpu_time': start.elapsed_time(end),
              'input_mem': input_mem,
              'model_mem': model_mem,
              'total_mem': total_mem}
    if use_dtr:
        result['cuda_time'] = cuda_time
    else:
        result['cuda_time'] = -1.0

    return result


def timing_loop(model_name, i, config, use_dtr,
                specific_params, writer, trial_run=False, trial_run_outfile=None):
    dry_run = config['dry_run']
    measurements = []
    print(f'Running {model_name} : {specific_params}')

    # remove any logs hanging around (so we only have to look for one)
    delete_logs()

    # we only save logs for the final input on DTR
    save_log = use_dtr and config['save_logs'] and i == config['n_inputs'] - 1
    if use_dtr:
        torch.toggle_log(False)

    use_cudnn = model_util.use_cudnn(model_name)
    with torch.backends.cudnn.flags(enabled=use_cudnn, benchmark=use_cudnn):
        criterion = model_util.get_criterion(model_name)
        produce_model, gen_input, run_model, teardown = model_util.prepare_model(model_name, specific_params['batch_size'], use_dtr=use_dtr)
        inp = gen_input(i, specific_params.get('extra_params', dict()))

        progress = tqdm(range(dry_run + config['n_reps']))
        for j in progress:
            progress.set_description(f'Rep [{j}]' + '' if j > dry_run else f'Dry run [{j}]')
            gc.collect()
            # Annotate where the final run starts in the log
            if save_log and j == dry_run + config['n_reps'] - 1:
                torch.toggle_log(config['save_logs'])
                torch.annotate_log('START')
 
            res = run_single_measurement(model_name, produce_model, run_model,
                                            teardown, inp, criterion, extra_params=specific_params.get('extra_params', dict()), use_dtr=use_dtr)
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
        save_trial_log(config['log_dest'], model_name, specific_params)

    # clean up after ourselves
    delete_logs()

    # do all the writing after the trial is over
    for j in range(len(measurements)):
        data = measurements[j]
        # do unit conversions now: times in ms,
        # memory in MB
        writer.writerow({
            'time': data['time']*1e3,
            # pytorch's cuda elapsed time is already in ms
            'gpu_time': float(data['gpu_time']),
            # 'cuda_time' : float(data['cuda_time']) * 1e-6,
            'input_mem': data['input_mem']*1e-6,
            'model_mem': data['model_mem']*1e-6,
            'total_mem': data['total_mem']*1e-6,
            'rep': j - dry_run,
            'input': i,
            **specific_params
         })


def main(config_dir, experiment_mode, model_name, input_idx, params_file, out_file,
         trial_run=False, trial_run_outfile=None):
    config, msg = validate(config_dir)
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
        timing_loop(model_name, i, config, use_dtr, specific_params, writer)


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'experiment_mode',
                'model_name', 'input_idx', 'params_file',
                'out_file', 'trial_run', 'trial_run_outfile')
