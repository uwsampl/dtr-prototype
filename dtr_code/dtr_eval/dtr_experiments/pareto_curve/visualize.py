import matplotlib
import os
import math
import numpy as np
import datetime
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import invoke_main, render_exception, write_status, write_json, sort_data, prepare_out_file
from validate_config import validate

name_dict = {
        'resnet32' : 'ResNet32',
        'lstm': 'LSTM',
        'treelstm': 'TreeLSTM',
        'unet' : 'UNet',
        'densenet100' : 'DenseNet100',
        'resnet1202' : 'ResNet1202'
    }
color_scheme = {
    'resnet32' : 'g',
    'lstm': 'r',
    'treelstm': 'c',
    'unet' : 'y',
    'densenet100' : 'm',
    'resnet1202' : 'b'
}
marker_scheme = {
    'resnet32' : '*',
    'lstm': 'o',
    'treelstm': 'v',
    'unet' : '^',
    'densenet100' : 's',
    'resnet1202' : '*'
}

def fill_data(data_dict, stat):
    exp_params = stat['specific_params']
    batch_size = exp_params['batch_size']
    exp_type = exp_params['type']
    exp_kind = exp_params.get('kind')

    def get_data():
        return {
                'cpu_time' : np.median([entry['time']['mean'] for entry in stat['summary']]),
                'gpu_time' : np.median([entry['gpu_time']['mean'] for entry in stat['summary']]),
                'mem'      : np.median([entry['total_mem']['mean'] for entry in stat['summary']]),
                'memory_budget' : exp_params.get('memory_budget', -1.0),
                'ratio' : exp_params.get('ratio', -1.0),
                'error' : False
            }

    if exp_type == 'baseline' and batch_size not in data_dict:
        if stat['summary'] == 'error':
            return data_dict
        data_dict[batch_size] = get_data()
        return data_dict
    if exp_type != 'baseline' and exp_kind in ('ratio', 'fixed'):
        if batch_size not in data_dict:
            data_dict[batch_size] = {}
        if exp_kind is not None and exp_kind not in data_dict[batch_size]:
            data_dict[batch_size][exp_kind] = []
        if stat['summary'] != 'error':
            data_dict[batch_size][exp_kind].append(get_data())
        else:
            data_dict[batch_size][exp_kind].append({
                'memory_budget' : exp_params.get('memory_budget', -1.0),
                'ratio' : exp_params.get('ratio', -1.0),
                'error' : True
            })

    return data_dict

def render_field(model_name, output_dir, title, filename, x_label, y_label, x_axis, baseline_entries, dtr_entries, failed_trials, suptitle=''):
    if not (dtr_entries or baseline_entries or failed_trials):
        return (True, 'nothing to render')
    file = prepare_out_file(output_dir, filename)
    try:
        # min_x = min(*(x_axis + failed_trials))
        # max_x = max(*(x_axis + failed_trials))
        ax = plt.gca()
        if dtr_entries:
            lin, = ax.plot(x_axis, dtr_entries, color=color_scheme.get(model_name, 'black'), linewidth=4)
            mk,  = ax.plot(x_axis, dtr_entries, label=name_dict.get(model_name, model_name), linewidth=4, marker=marker_scheme.get(model_name, '+'), ms=12, alpha=.6, color=color_scheme.get(model_name, 'black'))
            ax.legend([(lin, mk)], ['merged'])
        # if baseline_entries:
        #     plt.hlines(y=baseline_entries[0], xmin=min_x, xmax=max_x, linewidth=3, label='Baseline', color='blue', linestyles='dashed')

        if failed_trials:
            plt.axvline(x=max(failed_trials), color=color_scheme.get(model_name, 'black'), linestyle='dashed')

        # fig = plt.legend().figure
        # fig.savefig(file)
        return (True, 'success')
    except Exception as e:
        raise e
        return (False, 'Exception encountered while rendering graph: {}'.format(render_exception(e)))

def render_fixed(model_name, output_dir, x_axis, dtr_entries, failed_trials):
    if not (dtr_entries or failed_trials):
        return (True, 'nothing to render')
    filename = prepare_out_file(output_dir, f'{name_dict.get(model_name, model_name)}-fixed-gpu-time.png')
    try:
        plt.clf()
        plt.style.use('seaborn-paper')
        plt.rcParams["font.size"] = 30
        fig = plt.figure()
        fig.add_subplot(111, frameon=False)
        fig.set_size_inches(12, 7)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('Memory Budget (MB)', fontsize=15, labelpad=10)
        plt.ylabel(r'Compute Time (ms)', fontsize=15, labelpad=10)
        plt.title(f'{name_dict.get(model_name, model_name)} GPU Time', fontsize=18)
        plt.grid(True)

        ax = plt.gca()
        if dtr_entries:
            lin, = ax.plot(x_axis, dtr_entries, color=color_scheme.get(model_name, 'black'), linewidth=4)
            mk,  = ax.plot(x_axis, dtr_entries, label=name_dict.get(model_name, model_name), linewidth=4, marker=marker_scheme.get(model_name, '+'), ms=12, alpha=.6, color=color_scheme.get(model_name, 'black'))
            ax.legend([(lin, mk)], ['merged'])

        if failed_trials:
            plt.axvline(x=max(failed_trials), color=color_scheme.get(model_name, 'black'), linestyle='dashed')
        
        plt.legend(
                bbox_to_anchor=(0.5,0.01), 
                loc='lower center',
                bbox_transform=fig.transFigure, 
                ncol=7, 
                borderaxespad=0,
                prop={'size': 15}
            )
        plt.tight_layout()
        plt.savefig(filename, bbox_inches = 'tight')
        return (True, 'success')
    except Exception as e:
        raise e
        return (False, render_exception(e))


def render_time_comparison(model, batch_size, exp_kind, baseline_data, dtr_data, output_dir):
    # filename : model-batch_size-exp_kind-[time|gpu_time|mem]
    filename_template= f'{str(datetime.datetime.now()).replace(" ", "-")}-{model}-{batch_size}-{exp_kind}-' + '{}' + '.png'
    if exp_kind == 'ratio':
        comp = lambda k: k['ratio']
    elif exp_kind == 'fixed':
        comp = lambda k: k['memory_budget']
    
    sorted_entries = sorted(dtr_data, key=comp)
    dtr_entries = list(filter(lambda datum: not datum['error'], sorted_entries))
    failed_trials = list(map(comp, filter(lambda datum: datum['error'], sorted_entries)))
    x_axis = list(map(comp, dtr_entries))

    get_title = {
        'cpu_time' : f'{model} CPU Time Comparison',
        'gpu_time' : f'{model} GPU Time Comparison',
    }

    x_label = 'Memory Budget (Ratio)' if exp_kind == 'ratio' else 'Memory Budget (MB)'

    for field in ('gpu_time', ):
        dtr_stat = list(map(lambda k: k[field], dtr_entries))
        if baseline_data is not None:
            slow_down = list(map(lambda datnum: (datnum - baseline_data[field]) / baseline_data[field] + 1, dtr_stat))
        else:
            slow_down = []
        # baseline_stat = [] if baseline_data is None else [baseline_data[field]] * len(dtr_entries)
        # filename = filename_template.format(field)
        # success, msg = render_field(output_dir, get_title[field] + f' \n (batch size: {batch_size})', filename,
        #                         x_label, 'Time (ms)', x_axis, baseline_stat, dtr_stat, failed_trials)
        # if not success:
        #     return (False, msg)
        if exp_kind == 'ratio':
            xy = list(filter(lambda x: x[1] <= 4.0, zip(x_axis, slow_down)))
            x_axis = []
            slow_down = []
            for (x, y) in xy:
                x_axis.append(x)
                slow_down.append(y)
            filename = filename_template.format(field + '-slowdown')
            success, msg = render_field(model, output_dir, get_title[field] + f' \n (batch size: {batch_size})',
                                    filename, x_label, r'Overhead Slow Down (times)',
                                    x_axis, [], slow_down, [], f'Input Size: {batch_size}')
        elif exp_kind == 'fixed':
            data = list(map(lambda x: x[field], dtr_entries))
            success, msg = render_fixed(model, output_dir, x_axis, data, failed_trials)
        else:
            raise Exception(f'{exp_kind} is not a valid kind')

        if not success:
            return (False, msg)

    return (True, 'success')

def render_graph(config, data, output_dir):
    try:
        plt.style.use('seaborn-paper')
        plt.rcParams["font.size"] = 30
        fig = plt.figure()
        fig.add_subplot(111, frameon=False)
        fig.set_size_inches(12, 7)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('Memory Budget (Ratio)', fontsize=15, labelpad=10)
        plt.ylabel(r'Overhead Slow Down ($\times$)', fontsize=15, labelpad=10)
        plt.title('GPU Time Comparisons', fontsize=18)
        plt.grid(True)
        filename = prepare_out_file(output_dir, f'combined-comparison-ratio.png')

        metadata = {}
        for model in config['models']:
            dtr_dict = {}
            baseline_dict = {}
            stats = data[model]
            for stat in stats:
                if stat['specific_params']['type'] == 'baseline':
                    baseline_dict = fill_data(baseline_dict, stat)
                else:
                    dtr_dict = fill_data(dtr_dict, stat)
            
            metadata[model] = {
                'baseline' : baseline_dict,
                'dtr'      : dtr_dict
            }

            for batch_size in dtr_dict:
                baseline_data = baseline_dict.get(batch_size)
                for exp_kind in dtr_dict[batch_size]:
                    if exp_kind == 'ratio':
                        success, msg = render_time_comparison(model, batch_size, exp_kind, baseline_data, dtr_dict[batch_size][exp_kind], output_dir)
                        if not success:
                            return (False, msg)
        plt.hlines(y=1, xmin=0.0, xmax=1.0, linewidth=3, label='Baseline', color='blue', linestyles='dashed')
        plt.legend(
            bbox_to_anchor=(0.5,0.01), 
            loc='lower center',
            bbox_transform=fig.transFigure, 
            ncol=7, 
            borderaxespad=0,
            prop={'size': 15}
        )
        plt.tight_layout()
        plt.savefig(filename, bbox_inches = 'tight')

        for model in metadata:
            dtr_dict = metadata[model]['dtr']
            baseline_dict = metadata[model]['baseline']
            for batch_size in dtr_dict:
                baseline_data = baseline_dict.get(batch_size)
                for exp_kind in dtr_dict[batch_size]:
                    if exp_kind == 'fixed':
                        success, msg = render_time_comparison(model, batch_size, exp_kind, baseline_data, dtr_dict[batch_size][exp_kind], output_dir)
                        if not success:
                            return (False, msg)
        return (True, 'success')
    except Exception as e:
        raise e
        return (False, 'Exception encountered while rendering graphs: {}'.format(render_exception(e)))


def main(data_dir, config_dir, output_dir):
    try:
        config, msg = validate(config_dir)
        if config is None:
            write_status(output_dir, False, msg)
            return 1

        all_data = sort_data(data_dir)
        most_recent = all_data[-1]
        success, msg = render_graph(config, most_recent, output_dir)
        write_status(output_dir, success, msg)
    except Exception as e:
        write_status(output_dir, False, 'Exception encountered: ' + render_exception(e))
        return 1
    finally:
        plt.close()


if __name__ == '__main__':
    invoke_main(main, 'data_dir', 'config_dir', 'output_dir')
