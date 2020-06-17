"""
Implementation of core dashboard infrastructure
"""
import datetime
import os
import random
import subprocess
import time
import functools

from common import (check_file_exists, idemp_mkdir, invoke_main, get_timestamp,
                    prepare_out_file, read_json, write_json, read_config, validate_json, print_log)
from dashboard_info import DashboardInfo

def validate_status(dirname):
    return validate_json(dirname, 'success', 'message')


def attempt_parse_config(config_dir, target):
    """
    Returns the parsed config for the target (experiment or subsystem) if it exists.
    Returns None if the config is missing or could not be parsed.
    """
    conf_subdir = os.path.join(config_dir, target)
    if not check_file_exists(conf_subdir, 'config.json'):
        return None

    try:
        return read_json(conf_subdir, 'config.json')
    except Exception as e:
        return None


def check_present_and_executable(subdir, filenames):
    """
    Checks that all the files in the list are present in the subdirectory
    and are executable. Returns a list of any files in the list that are
    not present or not executable.
    """
    invalid = []
    for filename in filenames:
        path = os.path.join(subdir, filename)
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            invalid.append(filename)
    return invalid


def target_precheck(root_dir, configs_dir, target_name,
                    info_defaults, required_scripts):
    """
    Checks:
    1. That the target (subsys or experiment) config includes an 'active' field indicating whether to run it
    2. If the target is active, check that all required
    scripts are present and executable
    This function returns:
    1. a dict containing a 'status' field
    (boolean, true if all is preconfigured correctly) and a 'message' containing an
    explanation as a string if one is necessary
    2. A dict containing the target config's entries for
    each of the fields in info_defaults (uses the default
    if it's not specified)
    """
    target_conf = attempt_parse_config(configs_dir, target_name)
    if target_conf is None:
        return ({'success': False,
                 'message': 'config.json for {} is missing or fails to parse'.format(target_name)},
                None)

    update_fields = []
    target_info = {}
    for field, default in info_defaults.items():
        update_fields.append(field)
        target_info[field] = default

    for field in update_fields:
        if field in target_conf:
            target_info[field] = target_conf[field]

    # no need to check target subdirectory if it is not active
    if not target_conf['active']:
        return ({'success': True, 'message': 'Inactive'}, target_info)

    target_subdir = os.path.join(root_dir, target_name)
    if not os.path.exists(target_subdir):
        return ({'success': False,
                 'message': 'Script subdirectory for {} missing'.format(target_name)}, None)

    invalid_scripts = check_present_and_executable(target_subdir, required_scripts)
    if invalid_scripts:
        return ({
            'success': False,
            'message': 'Necessary files are missing from {} or not executable: {}'.format(
                target_subdir,
                ', '.join(invalid_scripts))
        },
                None)

    return ({'success': True, 'message': ''}, target_info)


def experiment_precheck(info, experiments_dir, exp_name):
    return target_precheck(
        experiments_dir, info.exp_configs, exp_name,
        {
            'active': False,
            'priority': 0,
            'rerun_setup': False
        },
        ['run.sh', 'analyze.sh', 'visualize.sh', 'summarize.sh'])


def has_setup(experiments_dir, exp_name):
    setup_path = os.path.join(experiments_dir, exp_name, 'setup.sh')
    return os.path.isfile(setup_path) and os.access(setup_path, os.X_OK)


def most_recent_experiment_update(experiments_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)
    git_list = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], cwd=exp_dir)
    files = git_list.decode('UTF-8').strip().split('\n')
    most_recent = None
    for f in files:
        raw_date = subprocess.check_output(['git', 'log', '-1', '--format=\"%ad\"', '--', f], cwd=exp_dir)
        date_str = raw_date.decode('UTF-8').strip(' \"\n')
        parsed = datetime.datetime.strptime(date_str, '%a %b %d %H:%M:%S %Y %z')
        if most_recent is None or most_recent < parsed:
            most_recent = parsed
    return most_recent


def last_setup_time(setup_dir, exp_name):
    marker_file = os.path.join(setup_dir, exp_name, '.last_setup')
    if os.path.isfile(marker_file):
        t = os.path.getmtime(marker_file)
        return time.localtime(t)
    return None


def should_setup(experiments_dir, setup_dir, exp_name):
    last_setup = last_setup_time(setup_dir, exp_name)
    if last_setup is None:
        return True

    most_recent = most_recent_experiment_update(experiments_dir, exp_name).timetuple()
    return most_recent > last_setup


def setup_experiment(info, experiments_dir, setup_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)
    exp_setup_dir = os.path.join(setup_dir, exp_name)

    # remove the existing setup dir before running the script again
    subprocess.call(['rm', '-rf', exp_setup_dir])
    idemp_mkdir(exp_setup_dir)

    subprocess.call([os.path.join(exp_dir, 'setup.sh'), info.exp_config_dir(exp_name),
                     exp_setup_dir], cwd=exp_dir)

    status = validate_status(exp_setup_dir)
    info.report_exp_status(exp_name, 'setup', status)

    # if setup succeeded, touch a marker file so we know what time to check for changes
    if status['success']:
        subprocess.call(['touch', '.last_setup'], cwd=exp_setup_dir)

    return status['success']


def copy_setup(experiments_dir, setup_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)
    exp_setup_dir = os.path.join(setup_dir, exp_name)
    subprocess.call(['/bin/cp', '-rf', os.path.join(exp_setup_dir, '.'), 'setup/'],
                    cwd=exp_dir)


def run_experiment(info, experiments_dir, tmp_data_dir, exp_name):

    to_local_time = lambda sec: time.asctime(time.localtime(sec))
    exp_dir = os.path.join(experiments_dir, exp_name)
    exp_conf = info.exp_config_dir(exp_name)

    # set up a temporary data directory for that experiment
    exp_data_dir = os.path.join(tmp_data_dir, exp_name)
    idemp_mkdir(exp_data_dir)

    # Mark the start and the end of an experiment
    start_time = time.time()
    start_msg = f'Experiment {exp_name} starts @ {to_local_time(start_time)}'
    print_log(start_msg)
    # run the run.sh file on the configs directory and the destination directory
    subprocess.call([os.path.join(exp_dir, 'run.sh'), exp_conf, exp_data_dir],
                    cwd=exp_dir)
    end_time = time.time()
    delta = datetime.timedelta(seconds=end_time - start_time)
    # collect the status file from the destination directory, copy to status dir
    status = validate_status(exp_data_dir)
    # show experiment status to terminal
    if status['success']:
        end_msg = f'Experiment {exp_name} ends @ {to_local_time(end_time)}\nTime Delta: {delta}'
        print_log(end_msg)
    else:
        print_log(f'*** {exp_name} FAILED ***\n*** Reason: {status["message"]} ***')
    # record start & end & duration of an experiment
    status['start_time'] = to_local_time(start_time)
    status['end_time'] = to_local_time(end_time)
    status['time_delta'] = str(delta)
    # not literally copying because validate may have produced a status that generated an error
    info.report_exp_status(exp_name, 'run', status)
    return status['success']

def get_timing_info(info, exp_name):
    '''
        Get the timing information of an experiment
        recorded in `run.json`.
    '''
    run_status = validate_json(info.exp_status_dir(exp_name), 
                                    'success',
                                    'start_time', 
                                    'end_time', 
                                    'time_delta', filename='run.json')
    # validate run.json data
    keys = run_status.keys()
    if keys and functools.reduce(lambda x, y: x and y, 
                                    map(lambda x: x in keys, 
                                       ('start_time', 'end_time', 'time_delta'))):
        rs_get = run_status.get
        return {
            'start_time' : rs_get('start_time'),
            'end_time'   : rs_get('end_time'),
            'time_delta' : rs_get('time_delta')
        }
    return {}

def analyze_experiment(info, experiments_dir, tmp_data_dir,
                       date_str, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)

    exp_data_dir = os.path.join(tmp_data_dir, exp_name)
    tmp_analysis_dir = os.path.join(exp_data_dir, 'analysis')
    idemp_mkdir(tmp_analysis_dir)

    analyzed_data_dir = info.exp_data_dir(exp_name)
    if not os.path.exists(analyzed_data_dir):
        idemp_mkdir(analyzed_data_dir)

    subprocess.call([os.path.join(exp_dir, 'analyze.sh'),
                     info.exp_config_dir(exp_name), exp_data_dir, tmp_analysis_dir],
                    cwd=exp_dir)

    status = validate_status(tmp_analysis_dir)

    # read the analyzed data, append a timestamp field, and copy over to the permanent data dir
    if status['success']:
        data_exists = check_file_exists(tmp_analysis_dir, 'data.json')
        if not data_exists:
            status = {'success': False, 'message': 'No data.json file produced by {}'.format(exp_name)}
        else:
            # collect data to dump to data_*.json
            dump_data = {
                'timestamp'  : date_str,
            }
            dump_data.update(read_json(tmp_analysis_dir, 'data.json'))
            # fetch time spent on the experiment
            dump_data.update(get_timing_info(info, exp_name))
            write_json(analyzed_data_dir, 'data_{}.json'.format(date_str), dump_data)
    
    info.report_exp_status(exp_name, 'analysis', status)
    return status['success']


def visualize_experiment(info, experiments_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)

    exp_graph_dir = info.exp_graph_dir(exp_name)
    subprocess.call([os.path.join(exp_dir, 'visualize.sh'),
                     info.exp_config_dir(exp_name),
                     info.exp_data_dir(exp_name), exp_graph_dir],
                    cwd=exp_dir)

    status = validate_status(exp_graph_dir)
    info.report_exp_status(exp_name, 'visualization', status)


def summary_valid(exp_summary_dir):
    """
    Checks that the experiment summary directory contains a summary.json
    file and that the summary.json file contains the required fields, title
    and value.
    """
    exists = check_file_exists(exp_summary_dir, 'summary.json')
    if not exists:
        return False
    summary = read_json(exp_summary_dir, 'summary.json')
    return 'title' in summary and 'value' in summary


def summarize_experiment(info, experiments_dir, exp_name):
    exp_dir = os.path.join(experiments_dir, exp_name)

    exp_summary_dir = info.exp_summary_dir(exp_name)
    subprocess.call([os.path.join(exp_dir, 'summarize.sh'),
                     info.exp_config_dir(exp_name), info.exp_data_dir(exp_name),
                     exp_summary_dir],
                    cwd=exp_dir)

    status = validate_status(exp_summary_dir)
    if status['success'] and not summary_valid(exp_summary_dir):
        status = {
            'success': False,
            'message': 'summary.json produced by {} is invalid'.format(exp_name)
        }
    info.report_exp_status(exp_name, 'summary', status)

def run_all_experiments(info, experiments_dir, setup_dir,
                        tmp_data_dir, data_archive,
                        time_str, randomize=True):
    """
    Handles logic for setting up and running all experiments.
    """
    exp_status = {}
    exp_confs = {}

    # do the walk of experiment configs, take account of which experiments are
    # either inactive or invalid
    for exp_name in info.all_present_experiments():
        precheck, exp_info = experiment_precheck(info, experiments_dir, exp_name)
        info.report_exp_status(exp_name, 'precheck', precheck)
        exp_status[exp_name] = 'active'
        exp_confs[exp_name] = exp_info
        if not precheck['success']:
            exp_status[exp_name] = 'failed'
            continue
        if not exp_info['active']:
            exp_status[exp_name] = 'inactive'

    active_exps = [exp for exp, status in exp_status.items() if status == 'active']

    # handle setup for all experiments that have it
    for exp in active_exps:
        if has_setup(experiments_dir, exp):
            # run setup if the most recent updated file is more recent
            # than the last setup run or if the flag to rerun is set
            if should_setup(experiments_dir, setup_dir, exp) or exp_confs[exp]['rerun_setup']:
                success = setup_experiment(info, experiments_dir, setup_dir, exp)
                if not success:
                    exp_status[exp] = 'failed'
                    continue
            # copy over the setup files regardless of whether we ran it this time
            copy_setup(experiments_dir, setup_dir, exp)

    # for each active experiment, run and generate data
    active_exps = [exp for exp, status in exp_status.items() if status == 'active']

    if randomize:
        random.shuffle(active_exps)
    else:
        # if experiment order is not random, sort by experiment priority,
        # with name as a tie-breaker. Since we want higher priority exps to
        # be first, we use -priority as the first element of the key
        active_exps.sort(key=lambda exp: (-exp_confs[exp]['priority'], exp))

    for exp in active_exps:
        success = run_experiment(info, experiments_dir, tmp_data_dir, exp)
        if not success:
            exp_status[exp] = 'failed'

    # for each active experiment not yet eliminated, run analysis
    active_exps = [exp for exp, status in exp_status.items() if status == 'active']
    for exp in active_exps:
        success = analyze_experiment(info, experiments_dir, tmp_data_dir,
                                     time_str, exp)
        if not success:
            exp_status[exp] = 'failed'

    # after analysis we can compress the data
    subprocess.call(['tar', '-zcf', data_archive, tmp_data_dir])
    subprocess.call(['rm', '-rf', tmp_data_dir])

    # for each experiment for which analysis succeeded, run visualization and summarization
    active_exps = [exp for exp, status in exp_status.items() if status == 'active']
    for exp in active_exps:
        visualize_experiment(info, experiments_dir, exp)
        summarize_experiment(info, experiments_dir, exp)


def subsystem_precheck(info, subsystem_dir, subsys_name):
    return target_precheck(
        subsystem_dir, info.subsys_configs, subsys_name,
        {
            'active': False,
            'priority': 0
        },
        ['run.sh'])


def run_subsystem(info, subsystem_dir, subsys_name):
    subsys_dir = os.path.join(subsystem_dir, subsys_name)
    subsys_output_dir = info.subsys_output_dir(subsys_name)
    idemp_mkdir(subsys_output_dir)

    # remove the old status if one is hanging around
    # (subsystem output dirs remain around between runs)
    if check_file_exists(subsys_output_dir, 'status.json'):
        subprocess.call(['rm', '-f', os.path.join(subsys_output_dir, 'status.json')])

    # run the run.sh file on the configs directory and the output directory
    subprocess.call([os.path.join(subsys_dir, 'run.sh'),
                     info.subsys_config_dir(subsys_name),
                     info.home_dir, subsys_output_dir],
                    cwd=subsys_dir)

    # collect the status file from the destination directory, copy to status dir
    status = validate_status(subsys_output_dir)
    # not literally copying because validate may have produced a status that generated an error
    info.report_subsys_status(subsys_name, 'run', status)
    return status['success']


def run_all_subsystems(info, subsystem_dir, time_str):
    """
    Handles logic for setting up and running all subsystems.
    """
    subsys_status = {}
    subsys_confs = {}

    # do the walk of subsys configs, take account of which are inactive or invalid
    for subsys_name in info.all_present_subsystems():
        precheck, subsys_info = subsystem_precheck(info, subsystem_dir, subsys_name)
        info.report_subsys_status(subsys_name, 'precheck', precheck)
        subsys_status[subsys_name] = 'active'
        subsys_confs[subsys_name] = subsys_info
        if not precheck['success']:
            subsys_status[subsys_name] = 'failed'
            continue
        if not subsys_info['active']:
            subsys_status[subsys_name] = 'inactive'

    active_subsys = [subsys for subsys, status in subsys_status.items() if status == 'active']

    # high priority = go earlier, so we prioritize with negative priority, with the name as tiebreaker
    active_subsys.sort(key=lambda subsys: (-subsys_confs[subsys]['priority'], subsys))

    for subsys in active_subsys:
        success = run_subsystem(info, subsystem_dir, subsys)


def main(home_dir, experiments_dir, subsystem_dir):
    """
    Home directory: Where config info for experiments, etc., is
    Experiments directory: Where experiment implementations are
    Both should be given as absolute directories
    """
    time_str = get_timestamp()
    home_dir = os.path.abspath(home_dir)
    experiments_dir = os.path.abspath(experiments_dir)
    subsystem_dir = os.path.abspath(subsystem_dir)

    if not check_file_exists(home_dir, 'config.json'):
        print('Dashboard config (config.json) is missing in {}'.format(home_dir))
        return 1
    dash_config = read_json(home_dir, 'config.json')

    # must expand all tildes in the config to avoid future errors
    for path_field in ['tmp_data_dir', 'setup_dir', 'backup_dir']:
        dash_config[path_field] = os.path.expanduser(dash_config[path_field])

    tmp_data_dir = os.path.join(dash_config['tmp_data_dir'], 'benchmarks_' + time_str)
    data_archive = os.path.join(dash_config['tmp_data_dir'], 'benchmarks_' + time_str + '_data.tar.gz')
    setup_dir = dash_config['setup_dir']
    backup_archive = os.path.join(dash_config['backup_dir'], 'dashboard_' + time_str + '.tar.gz')
    idemp_mkdir(tmp_data_dir)
    idemp_mkdir(os.path.dirname(backup_archive))
    idemp_mkdir(setup_dir)

    info = DashboardInfo(home_dir)

    # make a backup of the previous dashboard files if they exist
    if os.path.exists(home_dir):
        subprocess.call(['tar', '-zcf', backup_archive, home_dir])

    # directories whose contents should not change between runs of the dashboard
    persistent_dirs = {info.exp_data,
                       info.exp_configs,
                       info.subsys_configs,
                       info.subsys_output}
    all_dashboard_dirs = info.all_experiment_dirs() + info.all_subsystem_dirs()

    # instantiate necessary dashboard dirs and clean any that should be empty
    for dashboard_dir in all_dashboard_dirs:
        if dashboard_dir not in persistent_dirs:
            subprocess.call(['rm', '-rf', dashboard_dir])
        idemp_mkdir(dashboard_dir)

    randomize_exps = True
    if 'randomize' in dash_config:
        randomize_exps = dash_config['randomize']

    run_all_experiments(info, experiments_dir, setup_dir,
                        tmp_data_dir, data_archive,
                        time_str, randomize=randomize_exps)

    run_all_subsystems(info, subsystem_dir, time_str)


if __name__ == '__main__':
    invoke_main(main, 'home_dir', 'experiments_dir', 'subsystem_dir')
