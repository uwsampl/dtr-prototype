"""
To avoid any issues of memory hanging around between inputs,
we run each input as a separate process.

A little ugly but effective
"""
import glob
import os
import random
import math
import time

from queue import Queue as LocalQueue # contrast with mp.Queue
import multiprocessing as mp

from common import invoke_main, read_json, write_json, prepare_out_file, check_file_exists

from validate_config import validate_trials_config

def extend_simrd_config(dest_dir, sim_conf_filename, model_name, specific_params, log_name):
    import model_util
    if not check_file_exists(dest_dir, sim_conf_filename):
        prepare_out_file(dest_dir, sim_conf_filename)
        write_json(dest_dir, sim_conf_filename, dict())

    conf = read_json(dest_dir, sim_conf_filename)
    if model_name not in conf:
        conf[model_name] = []
    name = model_util.format_model_name(model_name, specific_params)
    conf[model_name].append({
        'name': name,
        'title': name,
        'desc': model_util.format_input_description(model_name, specific_params),
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
    if sim_conf_filename is not None:
        extend_simrd_config(dest_dir, sim_conf_filename, model_name, specific_params, filename)


def delete_logs():
    for log in glob.glob(os.path.join(os.getcwd(), '*.log')):
        os.remove(log)


def report_results(model_name, i, config, specific_params, num_retries,
                   out_file, use_dtr, trial_run, trial_run_outfile, results_queue):
    """
    Given a queue of results, do all the necessary reporting
    """
    measurements = []
    while not results_queue.empty():
        measurements.append(results_queue.get())

    # all we care about for a trial run is max memory usage
    if trial_run:
        write_json(os.getcwd(), trial_run_outfile, {
            'mem' : max(map(lambda data: data['total_mem'], measurements))
        })
        return

    memory_budget = specific_params.get('memory_budget', -1)
    dry_run = config['dry_run']
    save_log = use_dtr and specific_params.get('save_logs', config['save_logs']) and i == config['n_inputs'] - 1

    if save_log:
        save_trial_log(config['log_dest'], config.get('simrd_config', None),
                       model_name,
                       specific_params,
                       is_baseline=(memory_budget == -1))

    # clean up after ourselves
    delete_logs()

    with open(out_file, 'a', newline='') as csvfile:
        from pt_trial_util import create_csv_writer
        writer = create_csv_writer(csvfile, specific_params)
        for j in range(len(measurements)):
            data = measurements[j]
            # do unit conversions now: times in ms,
            # memory in MB
            entry = {
                'num_retries': num_retries,
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
            }
            writer.writerow(entry)


# TODO: Since this will be a separate process, we don't need
# run_torch_trial as a whole to be a separate script anymore
def run_measurements(config, specific_params, i, model_name,
                     dry_run, n_reps, use_dtr,
                     results_queue, heartbeat_queue):
    """
    Sets up PyTorch and runs the specified number of measurements,
    placing them in the given queue.

    Handles all PT setup so it can be spun off into a separate process
    """
    import torch
    import gc

    import math
    import signal

    from validate_config import validate_trials_config
    from pt_trial_util import create_csv_writer
    from tqdm import tqdm
    import model_util

    def set_sampling(specific_params):
        """
        Process the sampling cutoff if it's set (don't do random sampling
        if sampling is on, the budget is set, and the budget is
        below the specified sampling threshold)
        """
        memory_budget = int(specific_params['memory_budget'])
        sampling_flag = specific_params['use_sampling']
        sampling_cutoff = -1
        if 'no_sampling_below_budget' in specific_params:
            sampling_cutoff = int(specific_params['no_sampling_below_budget'])

        # If sampling is off anyway, there's no need to check.
        # If there is no cutoff or no budget set, we can't check.
        if (not sampling_flag or sampling_cutoff == -1 or memory_budget == -1):
            torch.toggle_sampling(sampling_flag)
            return

        # otherwise, check the threshold (don't sample if the threshold is met)
        threshold_met = (memory_budget <= sampling_cutoff)
        torch.toggle_sampling(not threshold_met)

    def run_single_measurement(model_name, produce_model, run_model, teardown, inp,
                               criterion, extra_params, use_dtr, use_profiling):
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

    def timing_loop(model_name, i, dry_run, n_reps,
                    config, use_dtr,
                    specific_params, extra_params,
                    results_queue, heartbeat_queue):
        measurements = []
        print(f'Running {model_name} : {specific_params}')

        # remove any logs hanging around (so we only have to look for one)
        delete_logs()

        # we only save logs for the final input on DTR
        save_log = use_dtr and specific_params.get('save_logs', config['save_logs']) and i == config['n_inputs'] - 1
        if use_dtr:
            torch.toggle_log(False)

        batch_size = specific_params['batch_size']
        use_profiling = use_dtr and specific_params.get('use_profiling', False)
        use_cudnn = model_util.use_cudnn(model_name)

        with torch.backends.cudnn.flags(enabled=use_cudnn, benchmark=use_cudnn):
            produce_model, gen_input, run_model, teardown = model_util.prepare_model(model_name,
                                                                                     batch_size,
                                                                                     use_dtr=use_dtr)
            criterion = model_util.get_criterion(model_name)
            inp = gen_input(i, extra_params)

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

                try:
                    res = run_single_measurement(model_name, produce_model, run_model, teardown,
                                                 inp, criterion, extra_params, use_dtr, use_profiling)
                except RuntimeError as e:
                    heartbeat_queue.put((False, 0))
                    raise e
                heartbeat_queue.put((True, res["time"]))
                if j >= dry_run:
                    results_queue.put(res)

    extra_params = specific_params.get('extra_params', {})
    if config['set_seed']:
        torch.manual_seed(config['seed'] + i)
        random.seed(config['seed'] + i)

    # DTR-specific setup
    assert 'batch_size' in specific_params
    if use_dtr:
        assert 'memory_budget' in specific_params
        memory_budget = int(specific_params['memory_budget'])
        if memory_budget > 0:
            print(f'Setting budget to {memory_budget}')
            torch.set_memory_budget(memory_budget)

        if 'ignore_small_tensors' in specific_params:
            torch.toggle_ignore_small_tensors(specific_params['ignore_small_tensors'])

        if 'use_sampling' in specific_params:
            set_sampling(specific_params)

    timing_loop(model_name, i, dry_run, n_reps,
                config, use_dtr,
                specific_params, extra_params,
                results_queue, heartbeat_queue)


def main(config_dir, experiment_mode, model_name, input_idx, params_file, out_file,
         trial_run=False, trial_run_outfile=None):
    config, msg = validate_trials_config(config_dir)
    if config is None:
        print(msg)
        return 1

    use_dtr = (experiment_mode == 'dtr')
    i = int(input_idx)
    is_trial = (trial_run == 'True')

    cwd = os.getcwd()
    specific_params = read_json(cwd, params_file)

    dry_run = config["dry_run"]
    n_reps = config["n_reps"]

    # TODO: this is very messy and we should make the setup nicer
    extra_params = specific_params.get('extra_params', {})
    retry_on_error = extra_params.get('retry_on_error', False)
    attempt_timeout = extra_params.get('attempt_timeout', 15)
    max_retries = extra_params.get('max_retries', 10)

    num_retries = 0
    if not retry_on_error:
        results_queue = LocalQueue()
        heartbeat_queue = LocalQueue()
        run_measurements(config, specific_params, i, model_name, dry_run, n_reps, use_dtr,
                         results_queue, heartbeat_queue)
    else:
        results_queue = mp.Queue()
        heartbeat_queue = mp.Queue()
        remaining_reps = n_reps
        for attempt in range(max_retries):
            proc = mp.Process(target=run_measurements,
                              args=(config, specific_params, i,
                                    model_name,
                                    dry_run, remaining_reps,
                                    use_dtr, results_queue, heartbeat_queue))
            proc.start()
            num_heartbeats = dry_run + remaining_reps
            last_timeout = attempt_timeout

            # TODO: clean this up
            encountered_error = False
            for b in range(num_heartbeats):
                try:
                    (success, last_time) = heartbeat_queue.get(block=True, timeout=last_timeout)
                    if not success:
                        print("Error in attempt")
                        encountered_error = True
                        break
                    last_timeout = math.ceil(2*last_time)
                except Exception as e:
                    print("Attempt timed out")
                    encountered_error = True
                    break
            if not encountered_error:
                break

            if proc.is_alive():
                proc.terminate()
            num_retries += 1

            successful_trials = (b+1) - dry_run
            if successful_trials > 0:
                remaining_reps -= successful_trials

        if num_retries == max_retries:
            raise RuntimeError("Used the max number of retries")

    report_results(model_name, i, config, specific_params, num_retries,
                   out_file, use_dtr, is_trial, trial_run_outfile, results_queue)

if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'experiment_mode',
                'model_name', 'input_idx', 'params_file',
                'out_file', 'trial_run', 'trial_run_outfile')
