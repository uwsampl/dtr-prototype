from common import invoke_main, render_exception, write_status, write_json

from validate_config import validate
from pt_trial_util import parse_commands, unfold_settings, parse_data_file

def main(data_dir, config_dir, output_dir):
    try:
        config, msg = validate(config_dir)
        if config is None:
            write_status(output_dir, False, msg)
            return 1

        summary = {}
        for model in sorted(config['models']):
            summary[model] = []
            # the script will not be run if there is an error
            cmd_id = 0
            for _, _, exp_config in parse_commands(model, config):
                for combo in unfold_settings(exp_config):
                    stats, msg = parse_data_file(exp_config['type'], model, config, combo, data_dir, cmd_id)
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
