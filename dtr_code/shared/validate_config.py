"""Checks that experiment config is valid and pre-populates default values."""
from common import read_config

from config_util import check_config, bool_cond, non_negative_cond, string_cond

MODELS = {
    # CIFAR resnets
    'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202',
    # Torchvision resnets
    'tv_resnet18', 'tv_resnet34', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152',
    # Torchvision densenets
    'tv_densenet121', 'tv_densenet161', 'tv_densenet169', 'tv_densenet201',
    'inceptionv3', 'inceptionv4',
    'unet', 'lstm_encoder', 'lstm', 'gru', 'gru_encoder', 'treelstm', 'treelstm_old',
    'transformer', 'transformer_encoder',
    'unroll_gan',
    # DenseNet-BC
    'densenet100'
}

def validate_simrd_config(config_dir):
    config = read_config(config_dir)
    return check_config(config, {
            "log_dest": "~/dtr_nightly_logs",
            "save_logs" : True,
            "simrd_config" : "sim_conf.json",
            "models": {
                "lstm",
                "resnet32",
                "treelstm",
                "densenet100",
                "unet",
                "resnet1202"
            },
            "simrd_experiments" : {
                "pareto",
                "ablation",
                "banishing"
            },
            "dtr_torch_cmd": "~/dtr_venv/bin/python3",
            "dry_run": 10,
            "n_inputs": 1,
            "n_reps": 50,
        }, {
            'models': MODELS
        }, {
            'dtr_torch_cmd': string_cond(),
            'dry_run': non_negative_cond(),
            'n_inputs': non_negative_cond(),
            'n_times_per_input': non_negative_cond(),
            'set_seed': bool_cond(),
            'seed': non_negative_cond(),
            'save_logs': bool_cond(),
            'log_dest': string_cond()
        })

def validate_trials_config(config_dir):
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
            'save_logs': False,
            'log_dest': '/dev/null',
            'report_errors': False,
            'set_seed': False,
            'seed': 0
        },
        {
            'models': MODELS,
            'methods': {'baseline', 'dtr'}
        },
        {
            'batch_size': non_negative_cond(),
            'dtr_torch_cmd': string_cond(),
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
