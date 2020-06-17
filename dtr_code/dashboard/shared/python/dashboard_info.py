"""
Centralized logic for dashboard file locations and other
high-level task information.
"""
from enum import Enum
import os

from common import (check_file_exists, read_json,
                    write_json, read_config)

def _check_stage_status(target_status_dir, stage_name):
    filename = '{}.json'.format(stage_name)
    if not check_file_exists(target_status_dir, filename):
        return {'success': False, 'message': '{} stage status missing'.format(stage_name)}

    try:
        return read_json(target_status_dir, filename)
    except:
        return {'success': False, 'message': 'Failed to parse {} stage status'.format(stage_name)}


def _report_stage_status(target_status_dir, stage_name, status):
    filename = '{}.json'.format(stage_name)
    write_json(target_status_dir, filename, status)


def _yield_subdir_names(base_dir):
    for subdir, _, _ in os.walk(base_dir):
        if subdir == base_dir:
            continue
        yield os.path.basename(subdir)


class SystemType(Enum):
    exp = 1
    subsys = 2


class InfoType(Enum):
    config = 1
    results = 2


class DashboardInfo:
    """
    Class that stores locations of important dashboard files
    and handles reasoning about file locations in the dashboard.
    Keeps the following directories:
    home_dir: (the one passed)
    exp_configs: (home)/config/experiments
    subsys_configs: (home)/config/subsystem
    exp_results: (home)/results/experiments
    exp_statuses: (home)/results/experiments/status
    exp_data: (home)/results/experiments/data
    exp_graphs: (home)/results/experiments/graph
    exp_summaries: (home)/results/experiments/summary
    subsys_results: (home)/results/subsystem
    subsys_statuses: (home)/results/subsystem/status
    subsys_output: (home)/results/subsystem/output
    Accessors:
    exp_{field}_dir(exp_name): (home)/(field path)/exp_name
    subsys_{field}_dir(subsys_name): (home)/(field path)/subsys_name
    For example, exp_config_dir(exp_name) returns (home)/config/experiments/exp_name
    and subsys_output_dir(subsys_name) returns (home)/results/subsystem/output/subsys_name
    """
    def __init__(self, home_dir):
        self.home_dir = home_dir
        config_dir = os.path.join(self.home_dir, 'config')
        results_dir = os.path.join(home_dir, 'results')

        # better to generate the fields and accessors than have a full file of boilerplate
        # info type, system type, singular, plural
        dashboard_fields = [
            (InfoType.config,  SystemType.exp,    'config',    'configs'),
            (InfoType.results, SystemType.exp,    'data',      'data'),
            (InfoType.results, SystemType.exp,    'status',    'statuses'),
            (InfoType.results, SystemType.exp,    'graph',     'graphs'),
            (InfoType.results, SystemType.exp,    'summary',   'summaries'),
            (InfoType.config,  SystemType.subsys, 'config',    'configs'),
            (InfoType.results, SystemType.subsys, 'status',    'statuses'),
            (InfoType.results, SystemType.subsys, 'output',    'output'),
            (InfoType.results, SystemType.subsys, 'telemetry', 'telemetry'),
        ]

        # we need to have a function return the lambda for proper closure behavior
        def gen_accessor(base_dir):
            return lambda target_name: os.path.join(base_dir, target_name)

        for (info_type, system_type, singular_name, plural_name) in dashboard_fields:
            base_dir = config_dir if info_type == InfoType.config else results_dir
            abbrev = 'exp' if system_type == SystemType.exp else 'subsys'
            sys_designation = 'experiments' if system_type == SystemType.exp else 'subsystem'
            subdir = os.path.join(base_dir, sys_designation)
            if info_type == InfoType.results:
                subdir = os.path.join(subdir, singular_name)
            setattr(self, '{}_{}'.format(abbrev, plural_name), subdir)
            setattr(self, '{}_{}_dir'.format(abbrev, singular_name), gen_accessor(subdir))


    def all_experiment_dirs(self):
        return [
            self.exp_configs,
            self.exp_data, self.exp_graphs, self.exp_summaries,
            self.exp_statuses
        ]

    def all_subsystem_dirs(self):
        return [
            self.subsys_configs,
            self.subsys_output, self.subsys_statuses
        ]


    def exp_config_valid(self, exp_name):
        return _check_stage_status(self.exp_status_dir(exp_name), 'precheck')['success']


    def subsys_config_valid(self, subsys_name):
        return _check_stage_status(self.subsys_status_dir(subsys_name), 'precheck')['success']


    def read_exp_summary(self, exp_name):
        return read_json(self.exp_summary_dir(exp_name), 'summary.json')


    def read_exp_config(self, exp_name):
        return read_config(self.exp_config_dir(exp_name))


    def read_subsys_config(self, subsys_name):
        return read_config(self.subsys_config_dir(subsys_name))

    def exp_cpu_telemetry(self, exp_name):
        return os.path.join(self.subsys_telemetry_dir(exp_name), 'cpu')

    def exp_gpu_telemetry(self, exp_name):
        return os.path.join(self.subsys_telemetry_dir(exp_name), 'gpu')

    def exp_active(self, exp_name):
        return self.exp_config_valid(exp_name) and self.read_exp_config(exp_name)['active']


    def subsys_active(self, subsys_name):
        return self.subsys_config_valid(subsys_name) and self.read_subsys_config(subsys_name)['active']


    def exp_stage_statuses(self, exp_name):
        ret = {'precheck': self.exp_stage_status(exp_name, 'precheck')}

        if not ret['precheck']['success'] or not self.exp_active(exp_name):
            return ret

        # setup is the only optional stage
        if check_file_exists(self.exp_status_dir(exp_name), 'setup.json'):
            ret['setup'] = self.exp_stage_status(exp_name, 'setup')
            if not ret['setup']['success']:
                return ret

        for stage in ['run', 'analysis', 'summary', 'visualization']:
            ret[stage] = self.exp_stage_status(exp_name, stage)
            if not ret[stage]['success']:
                break

        return ret


    def subsys_stage_statuses(self, subsys_name):
        ret = {'precheck': self.subsys_stage_status(subsys_name, 'precheck')}

        if not ret['precheck'] or not self.subsys_active(subsys_name):
            return ret

        ret['run'] = self.subsys_stage_status(subsys_name, 'run')
        return ret


    def exp_stage_status(self, exp_name, stage):
        return _check_stage_status(self.exp_status_dir(exp_name), stage)


    def subsys_stage_status(self, subsys_name, stage):
        return _check_stage_status(self.subsys_status_dir(subsys_name), stage)


    def report_exp_status(self, exp_name, stage, status):
        return _report_stage_status(self.exp_status_dir(exp_name), stage, status)


    def report_subsys_status(self, subsys_name, stage, status):
        return _report_stage_status(self.subsys_status_dir(subsys_name), stage, status)


    def all_present_experiments(self):
        """
        Iterates through all subdirectories present in
        config/experiments.
        """
        return _yield_subdir_names(self.exp_configs)


    def all_present_subsystems(self):
        """
        Iterates through all subdirectories present in
        config/subsystem.
        """
        return _yield_subdir_names(self.subsys_configs)
