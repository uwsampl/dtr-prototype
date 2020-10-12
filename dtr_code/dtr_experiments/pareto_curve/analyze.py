from common import invoke_main, render_exception, write_status, write_json

from validate_config import validate_trials_config
from pt_trial_util import parse_commands, unfold_settings, parse_data_file

def main(data_dir, config_dir, output_dir):
    try:
        config, msg = validate_trials_config(config_dir)
        if config is None:
            write_status(output_dir, False, msg)
            return 1

        summary = {}

        baseline_dict = {}

        for model in sorted(config['models']):
            summary[model] = []
            baseline_dict[model] = {}
            # the script will not be run if there is an error
            cmd_id = 0
            for _, _, exp_config in parse_commands(model, config):
                baseline_params = None
                for specific_params in unfold_settings(exp_config):
                    batch_size = specific_params['batch_size']
                    if specific_params['type'] == 'baseline':
                        baseline_dict[model][batch_size] = {
                            'type': 'baseline',
                            'specific_params': specific_params,
                            'cmd_id': cmd_id
                        }

                    # if there is a corresponding baseline,
                    # let's match using the dict
                    baseline_params = None
                    if (batch_size in baseline_dict[model]
                        and specific_params['type'] != 'baseline'):
                        baseline_params = baseline_dict[model][batch_size]

                    stats, msg = parse_data_file(exp_config['type'], model, config,
                                                 specific_params, data_dir, cmd_id,
                                                 baseline_params=baseline_params)
                    if stats is None:
                        write_status(output_dir, False, msg)
                        return 1
                    stats['command_id'] = cmd_id
                    summary[model].append(stats)
                cmd_id += 1
        write_json(output_dir, 'data.json', summary)
        write_status(output_dir, True, 'success')
    except Exception as e:
        write_status(output_dir, False, render_exception(e))


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
