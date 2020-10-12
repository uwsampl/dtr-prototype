import os

from common import invoke_main, render_exception, write_status

from pt_trial_util import parse_commands, eval_command
from validate_config import validate_trials_config

def main(config_dir, output_dir):
    config, msg = validate_trials_config(config_dir)
    if config is None:
        write_status(output_dir, False, msg)
        return 1

    if not config['models']:
        write_status(output_dir, True, 'Nothing run')
        return 0

    for model in sorted(config['models']):
        cmd_id = 0
        for success, msg, processed_command in parse_commands(model, config):
            if not success:
                write_status(output_dir, False, msg)
                return 1
            else:
                print(f'Running on command: {model}: {processed_command}')
                success, msg = eval_command(model, processed_command, config, config_dir, output_dir, cmd_id)
                if not success:
                    write_status(output_dir, False, msg)
                    return 1
                cmd_id += 1

    write_status(output_dir, True, 'success')


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'output_dir')
