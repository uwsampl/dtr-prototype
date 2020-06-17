from common import (invoke_main, render_exception, sort_data,
                    write_status, write_summary)

from validate_config import validate
import numpy as np
from functools import reduce

def filter_params(specific_params):
    result = {
        'type' : specific_params['type'],
    }

    if specific_params.get('memory_budget') is not None and specific_params['memory_budget'] > 0:
        result['memory_budget'] = '{:.3g} MB'.format(specific_params['memory_budget'] * 1e-6)

    if result['type'] == 'dtr':
        if specific_params['kind'] == 'ratio':
            result['ratio'] = specific_params['ratio']

    return result


def summarize_results(stat):
    return (
        np.median([entry['time']['mean'] for entry in stat['summary']]),
        np.median([entry['gpu_time']['mean'] for entry in stat['summary']]),
        # memories are in MB
        np.median([entry['input_mem']['mean'] for entry in stat['summary']]),
        np.median([entry['model_mem']['mean'] for entry in stat['summary']]),
        np.median([entry['total_mem']['mean'] for entry in stat['summary']])
    )


def summarize(config, data):
    indent = ' ' * 3
    error_summary = '*ERRORS CAUGHT AT:*\n'
    summary = 'Key: median (Wall clock time (ms), GPU time (ms), input memory (MB), model memory (MB), and final memory (MB)) for each input\n'

    failed_models = {}

    for model in config['models']:
        model_data = data[model]
        result_by_settings = []
        for stat in model_data:
            if stat['summary'] == 'error':
                if model not in failed_models:
                    failed_models[model] = []
                result = filter_params(stat['specific_params'])
                result.update({'command_id' : stat['command_id']})
                if result not in failed_models[model]:
                    failed_models[model].append(result)
                continue

            configuration_str = '; '.join([
                indent + '_{}_: {}'.format(k, v)
                for k, v in filter_params(stat['specific_params']).items()
            ])
            summaries = ', '.join([
                '{:.3f}'.format(res) for res in summarize_results(stat)
            ])

            result_by_settings.append({
                'heading' : indent + 'Configuration:\n{}\n'.format(configuration_str),
                'summaries' : [indent*2 + '*Results*: {}\n'.format(summaries)]
            })

        if result_by_settings:
            summary += '*{}*:\n'.format(model)
            for results in result_by_settings:
                summary += results['heading']
                for line in results['summaries']:
                    summary += line
                summary += '\n'
    if failed_models:
        for model, settings in failed_models.items():
            error_summary += f'*{model}*:\n'
            for specific_params in settings:
                error_summary += (';' + indent).join(['_{}_: {}'.format(k, v) for k, v, in specific_params.items()]) + '\n'
            error_summary += '\n'
        return error_summary + '\n' + summary
    else:
        return summary


def main(data_dir, config_dir, output_dir):
    try:
        config, msg = validate(config_dir)
        if config is None:
            write_status(output_dir, False, msg)
            return 1

        all_data = sort_data(data_dir)
        most_recent = all_data[-1]

        summary = summarize(config, most_recent)
        write_summary(output_dir, 'Pareto Curve Trial', summary)
        write_status(output_dir, True, 'success')

    except Exception as e:
        write_status(output_dir, False, 'Exception encountered: ' + render_exception(e))
        return 1


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
