"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config

from config_util import check_config, bool_cond, non_negative_cond, string_cond

def validate(config_dir):
    """
    Reads config.json in the config_dir and prepopulates with default values.
    Ensures that all configured values are of the appropriate types.

    Returns (config, message to report if error). Returns None if something
    is wrong with the config it read.
    """
    config = read_config(config_dir)
    return check_config(
        config,
        {
            'dry_run': 8,
            'n_inputs': 3,
            'n_times_per_input': 100,
            'models': {'resnet32'},
            'batch_size': [8],
            'dtr_torch_cmd': 'python3',
            'methods': {'baseline'},
            'save_logs': False,
            'log_dest': '/dev/null',
            'report_errors': False,
            'set_seed': False,
            'seed': 0
        },
        {
            'models': {
                'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202',
                'densenet100',
                'unet', 'lstm_encoder', 'lstm', 'gru', 'gru_encoder', 'treelstm', 'treelstm_old', 'unroll_gan'},
            'methods': {'baseline', 'dtr'}
        },
        {
            'batch_size': non_negative_cond(),
            'dtr_torch_cmd': string_cond(),
            'methods': string_cond(),
            'dry_run': non_negative_cond(),
            'n_inputs': non_negative_cond(),
            'n_times_per_input': non_negative_cond(),
            'report_errors': bool_cond(),
            'set_seed': bool_cond(),
            'seed': non_negative_cond(),
            'save_logs': bool_cond(),
            'log_dest': string_cond()
        }
    )
