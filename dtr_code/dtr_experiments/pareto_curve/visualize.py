import matplotlib
import os
import math
import numpy as np
import datetime
matplotlib.use('Agg')
matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
import matplotlib.pyplot as plt
from common import invoke_main, render_exception, write_status, write_json, sort_data, prepare_out_file
from validate_config import validate_trials_config

NAME_DICT = {
    'resnet32' : r'\textbf{ResNet32}',
    'lstm': r'\textbf{\Huge LSTM}',
    'treelstm': r'\textbf{\huge TreeLSTM}',
    'unet' : r'\textbf{\huge UNet}',
    'tv_densenet121' : r'\textbf{\huge DenseNet121}',
    'resnet1202' : r'\textbf{\huge ResNet1202}',
    'transformer' : r'\textbf{\huge Transformer}',
    'inceptionv3' : 'InceptionV3',
    'inceptionv4' : r'\textbf{\huge InceptionV4}',
    'unroll_gan': r'\textbf{\huge Unrolled Gan}'
}
COLOR_SCHEME = {
    'resnet32' : 'g',
    'lstm': 'r',
    'treelstm': 'c',
    'unet' : 'y',
    'tv_densenet121' : 'm',
    'resnet1202' : 'b',
    'inceptionv3' : 'grey',
    'inceptionv4' : 'slateblue',
    'unroll_gan': 'magenta'
}
MARKER_SCHEME = {
    'resnet32' : '*',
    'lstm': 'o',
    'treelstm': 'v',
    'unet' : '^',
    'tv_densenet121' : 's',
    'resnet1202' : '*',
    'inceptionv3' : '+',
    'inceptionv4' : 'D'
}

# Generated color scheme
breakdown_color_scheme = {
    'dispatch_overhead' : 'black',
    'base_compute_time'  : 'gold',
    'remat_compute_time' : 'silver',
    'search_time'        : 'slategray',
    'cost_time'          : 'crimson',
    'sync_time'          : 'purple'
}
breakdown_namedict = {
    'dispatch_overhead': 'Unprofiled',
    'base_compute_time': 'Model Operator',
    'remat_compute_time':'Rematerialization',
    'search_time'      : 'Eviction Loop',
    'cost_time'        : 'Cost Compute',
    'sync_time'        : 'Cuda Synchronization Time'
}

input_sizes = {
    'treelstm':    r'Binary tree with depth 9, node size 640 $\times$ 1',
    'resnet1202':  r'224 $\times$ 224',
    'tv_densenet121': r'224 $\times$ 224',
    'unet':        r'416 $\times$ 608',
    'inceptionv4': r'299 $\times$ 299',
    'transformer': r'512 (Sequence Length)',
    'lstm':        'Input Dimision: 512,\nHidden Dimision: 1700,\nSequence Length 128',
    'unroll_gan':  r'60 Steps, 512$\times$ 512'
}

timed_keys = ['base_compute_time', 'cost_time', 'search_time', 'remat_compute_time']
used_keys = ['cpu_time', 'throughput'] + timed_keys


def fill_data(data_dict, stat):
    exp_params = stat['specific_params']
    batch_size = exp_params['batch_size']
    exp_type = exp_params['type']
    exp_kind = exp_params.get('kind')

    def get_data():
        ret = {
            'cpu_time' : np.median([entry['time']['mean'] for entry in stat['summary']]),
            'cpu_conf' : (np.median([entry['time']['mean_conf'][0] for entry in stat['summary']]),
                          np.median([entry['time']['mean_conf'][1] for entry in stat['summary']])),
            'gpu_time' : np.median([entry['gpu_time']['mean'] for entry in stat['summary']]),
            'gpu_conf' : (np.median([entry['gpu_time']['mean_conf'][0] for entry in stat['summary']]),
                          np.median([entry['gpu_time']['mean_conf'][1] for entry in stat['summary']])),
            'mem'      : np.median([entry['total_mem']['mean'] for entry in stat['summary']]),
            'memory_budget' : exp_params.get('memory_budget', np.median([entry['memory_budget']['mean'] for entry in stat['summary']])),
            'ratio' : exp_params.get('ratio', -1.0),
            'sync_time': np.median([entry['sync_time']['mean'] for entry in stat['summary']]),
            'throughput': np.median([entry['throughput']['mean'] if 'throughput' in entry else 0 for entry in stat['summary']]),
            'throughput_conf': (np.median([entry['throughput']['mean_conf'][0] if 'throughput' in entry else 0 for entry in stat['summary']]),
                                np.median([entry['throughput']['mean_conf'][1] if 'throughput' in entry else 0 for entry in stat['summary']])),
            'base_compute_time': np.median([entry['base_compute_time']['mean'] for entry in stat['summary']]),
            'remat_compute_time': np.median([entry['remat_compute_time']['mean'] for entry in stat['summary']]),
            'cost_time': np.median([entry['cost_time']['mean'] for entry in stat['summary']]),
            'search_time': np.median([entry['search_time']['mean'] for entry in stat['summary']]),
            'error' : False
        }
        if 'slowdown' in stat['summary'][0]:
            ret['slowdown'] = np.median([entry['slowdown']['mean'] for entry in stat['summary']])
            ret['slowdown_conf'] = (np.median([entry['slowdown']['mean_conf'][0] for entry in stat['summary']]),
                                    np.median([entry['slowdown']['mean_conf'][1] for entry in stat['summary']]))
        return ret

    if exp_type == 'baseline' and batch_size not in data_dict:
        if stat['summary'] == 'error':
            return data_dict
        data_dict[batch_size] = get_data()
        return data_dict
    if exp_type != 'baseline' and exp_kind in ('ratio', 'fixed', 'param_sweep'):
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


def render_errorbars(ax, x_axis, entries, confidence):
    upper = list(map(lambda x: x[1], confidence))
    lower = list(map(lambda x: x[0], confidence))
    lolims = [True] * len(upper)
    uplims = [False] * len(upper)
    ax.errorbar(x_axis, entries, yerr=upper, lolims=lolims, uplims=uplims)
    ax.errorbar(x_axis, entries, yerr=lower, lolims=lolims, uplims=uplims)


def render_field(model_name, output_dir, title, filename, x_label, y_label, x_axis,
                 baseline_entries, dtr_entries, failed_trials, confidence=None, suptitle=''):
    if not (dtr_entries or baseline_entries or failed_trials):
        return (True, 'nothing to render')
    file = prepare_out_file(output_dir, filename)
    try:
        # min_x = min(*(x_axis + failed_trials))
        # max_x = max(*(x_axis + failed_trials))
        ax = plt.gca()
        if dtr_entries:
            lin, = ax.plot(x_axis, dtr_entries, color=COLOR_SCHEME.get(model_name, 'black'), linewidth=4)
            mk,  = ax.plot(x_axis, dtr_entries, label=NAME_DICT.get(model_name, model_name),
                           linewidth=4, marker=MARKER_SCHEME.get(model_name, '+'), ms=12,
                           alpha=.6, color=COLOR_SCHEME.get(model_name, 'black'))
            if confidence:
                render_errorbars(ax, x_axis, dtr_entries, confidence)
            ax.legend([(lin, mk)], ['merged'])
        # if baseline_entries:
        #     plt.hlines(y=baseline_entries[0], xmin=min_x, xmax=max_x, linewidth=3,
        #                label='Baseline', color='blue', linestyles='dashed')

        if failed_trials:
            plt.axvline(x=max(failed_trials), color=COLOR_SCHEME.get(model_name, 'black'), linestyle='dashed')

        # fig = plt.legend().figure
        # fig.savefig(file)
        return (True, 'success')
    except Exception as e:
        raise e
        return (False, 'Exception encountered while rendering graph: {}'.format(render_exception(e)))


def render_fixed(ax, model_name, output_dir, x_axis, dtr_entries, baseline_data, failed_trials, batch_size=None, confidence=None):
    if not (dtr_entries or failed_trials):
        return (True, 'nothing to render')
    filename = prepare_out_file(output_dir, f'{NAME_DICT.get(model_name, model_name)}-fixed-gpu-time.png')
    try:
        # plt.style.use('seaborn-paper')
        # plt.rcParams["font.size"] = 30
        # fig = plt.figure()
        # fig.add_subplot(111, frameon=False)
        # fig.set_size_inches(12, 7)
        # plt.xticks(fontsize=13)
        # plt.yticks(fontsize=13)
        # plt.xlabel('Memory Budget (MB)', fontsize=15, labelpad=10)
        # plt.ylabel(r'Compute Time (ms)', fontsize=15, labelpad=10)
        # plt.title(f'{NAME_DICT.get(model_name, model_name)} GPU Time', fontsize=18)
        # plt.grid(True)

        # ax = plt.gca()
        width = 0.0
        all_axis = sorted(x_axis + failed_trials)
        ind = np.arange(len(all_axis) + 1)
        ind_index = dict(zip(all_axis, ind))
        ind_pos = dict([(ind[i], i) for i in range(len(ind))])
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(map(lambda x: f'{round(x * 1e-9, 1)}', all_axis + [baseline_data['mem'] * 1e+6]))

        ax.tick_params(axis='both', labelsize=20)

        filtered_entries = []

        if baseline_data and 'cpu_time' in baseline_data:
            for (x, datum) in zip(x_axis, dtr_entries):
                if not datum.get('error', False) and 'cpu_time' in datum and datum['cpu_time'] > 3 * baseline_data['cpu_time']:
                    failed_trials.append(x)
                    filtered_entries.append({key : 0 for key in datum.keys()})
                else:
                    filtered_entries.append(datum)

        dtr_entries = filtered_entries

        if failed_trials:
            for x in failed_trials:
                ax.axvline(x=ind_index[x], color='red', linestyle='dashed', label='OOM')
        new_ind = []
        for x in x_axis:
            new_ind.append(ind_index[x])
        new_ind.append(ind[-1])
        ind = np.array(new_ind)
        ax.grid(True, axis='y')
        ax.set_title(f'{NAME_DICT.get(model_name, model_name)} ({batch_size})\n{input_sizes.get(model_name, "")}', fontsize=15)

        for x in failed_trials:
            ax.bar(ind_index[x], 0)
        if dtr_entries:
            # lin, = ax.plot(x_axis, dtr_entries, color=COLOR_SCHEME.get(model_name, 'black'), linewidth=4)
            # mk,  = ax.plot(x_axis, dtr_entries, label=NAME_DICT.get(model_name, model_name),
            #               linewidth=4, marker=MARKER_SCHEME.get(model_name, '+'), ms=12,
            #               alpha=.6, color=COLOR_SCHEME.get(model_name, 'black'))
            data_collection = { key : [] for key in timed_keys }
            data_collection['dispatch_overhead'] = []
            for entry in dtr_entries:
                acc = 0
                for (k, v) in entry.items():
                    if k != 'cpu_time':
                        data_collection[k].append(v)
                        acc += v
                data_collection['dispatch_overhead'].append(entry['cpu_time'] - acc)

            acc = np.zeros(len(x_axis))
            for k in timed_keys + ['dispatch_overhead']:
                # print(ind[:-1], data_collection[k])
                ax.bar(ind[:-1], data_collection[k], label=breakdown_namedict.get(k, k),
                               color=breakdown_color_scheme.get(k, 'red'),
                                bottom=acc)
                acc = acc + data_collection[k]

            if baseline_data and 'cpu_time' in baseline_data:
                ax.bar([ind[-1]], baseline_data['cpu_time'], label='Unmodified\nPyTorch', color='blue')
            else:
                ax.bar([ind[-1]], 0, label='Unmodified PyTorch', color='blue')
                ax.axvline(ind[-1], color='red', linestyle='dashed', label='OOM')

            if confidence and False:
                render_errorbars(ax, x_axis, dtr_entries, confidence)

            ax.invert_xaxis()
            # ax.legend([(lin, mk)], ['merged'])

                # plt.legend(
        #         bbox_to_anchor=(0.5,0.01),
        #         loc='lower center',
        #         bbox_transform=fig.transFigure,
        #         ncol=7,
        #         borderaxespad=0,
        #         prop={'size': 15}
        #     )
        # plt.tight_layout()
        # plt.savefig(filename, bbox_inches = 'tight')
        return (True, 'success')
    except Exception as e:
        raise e
        return (False, render_exception(e))


def render_time_comparison(model, batch_size, exp_kind, dtr_data, baseline_data, output_dir, plt_ax=None):
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
    field = 'gpu_time'
    field_conf = 'gpu_conf'

    if exp_kind == 'ratio':
        dtr_stat = list(map(lambda k: k['slowdown'], dtr_entries))
        err = list(map(lambda k: k['slowdown_conf'], dtr_entries))
        points = list(filter(lambda x: x[1] <= 2.0, zip(x_axis, dtr_stat, err)))
        x_axis = []
        slow_down = []
        confidence = []
        for (x, y, conf) in points:
            x_axis.append(x)
            slow_down.append(y)
            confidence.append([conf[0] - y, conf[1] - y])
        filename = filename_template.format(field + '-slowdown')
        success, msg = render_field(model, output_dir, get_title[field] + f' \n (batch size: {batch_size})',
                                    filename, x_label, r'Overhead Slow Down (times)',
                                    x_axis, [], slow_down, [],
                                    confidence=confidence, suptitle=f'Input Size: {batch_size}')
    elif exp_kind == 'fixed':
        data = list(map(lambda x: {field : x[field]
                                    for field in ['cpu_time'] + timed_keys }, dtr_entries))
        err = list(map(lambda x: x['cpu_conf'], dtr_entries))
        # x_axis = list(map(lambda x: x['mem'] * 1e+6, dtr_entries))

        confidence = [(interval[0] - measurement, interval[1] - measurement)
                for interval, measurement in zip(err, list(map(lambda x: x['cpu_time'], dtr_entries)))]
        success, msg = render_fixed(plt_ax, model, output_dir, x_axis, data, baseline_data, failed_trials, confidence=confidence, batch_size=batch_size)
    else:
        raise Exception(f'{exp_kind} is not a valid kind')

    if not success:
        return (False, msg)

    return (True, 'success')

def traverse_field(metadata, kind, func, output_dir):
    """
    Given filled metadata, the `kind` of experiments data to process, a function that takes
    (model, batch size, DTR data dict, baseline data dict, output directory) and the output directory
    this function traverses each entry of the metadata with respect to the keys in DTR data dictionary (batch size)
    and calls `func` on each entry that records the data of experiments that have kind of `kind`.
    returns a tuple of status and message. This function will propagate the status of `func` if it fails in
    the middle of execution
    """
    for model in metadata.keys():
        dtr_dict = metadata[model]['dtr']
        baseline_dict = metadata[model]['baseline']
        # traverse wrt DTR data dict since we assume that DTR *fails less freqently than baseline
        # *: baseline will have more OOM error at large batch sizes and small budgets
        for batch_size in sorted(dtr_dict.keys()):
            for exp_kind in dtr_dict[batch_size]:
                if exp_kind == kind:
                    success, msg = func(model, batch_size, dtr_dict, baseline_dict, output_dir)
                    if not success:
                        return (False, msg)
    return (True, 'success')

def render_throughput_breakdown(metadata, output_dir):
    throughput_metadata = {}


    # Gather data to render
    # a mapping that has the type model -> exp_type -> batch_size -> data dict
    def get_throughput_metadata(model, batch_size, dtr_dict, baseline_dict, output_dir):
        if model not in throughput_metadata:
            throughput_metadata[model] = {'dtr' : {}, 'baseline' : {}}
        throughput_metadata[model]['dtr'][batch_size] = []
        for datum in dtr_dict[batch_size]['param_sweep']:
            throughput_metadata[model]['dtr'][batch_size].append({
                'memory_budget': datum.get('memory_budget', -1),
                'error': datum['error'],
                **{ key : datum.get(key) for key in used_keys }
            })

        if batch_size in baseline_dict:
            throughput_metadata[model]['baseline'][batch_size] = { key : baseline_dict[batch_size][key] for key in used_keys }
        else:
            throughput_metadata[model]['baseline'][batch_size] = { key : 0 for key in used_keys }
        return True, 'success'

    traverse_field(metadata, 'param_sweep', get_throughput_metadata, output_dir)

    flip = lambda f: lambda x: lambda y: f(y, x)

    # Plot throughput and time breakdown of a model
    def plot_model(model):
        filename = prepare_out_file(output_dir, f'throughput-comparison-{model}.png')
        plt.clf()
        plt.grid(True)
        plt.title(f'Throughput Comparison of {NAME_DICT.get(model, model)}')
        plt.xlabel('Batch Size', fontsize=15, labelpad=10)
        plt.ylabel('Throughput (Batch Size / Avg GPU Time (s))')
        num_batch_size = len(throughput_metadata[model]['dtr'].keys())
        baseline_data = metadata[model]['baseline']
        width = 0.15
        ind = np.arange(num_batch_size)
        x_axis = list(sorted(throughput_metadata[model]['dtr'].keys()))

        # Wish we had currying !!!
        # If baseline data does not contain a batch size, then we fill 0 into the data, since it means baseline failed (OOMed)
        baseline_data = list(map(flip(throughput_metadata[model]['baseline'].get)(0), x_axis))

        # Bar for baseline
        plt.bar(ind, [datum['throughput'] for datum in baseline_data], width, label='Baseline')
        dtr_data = {
            'throughput': {},
            'breakdown': {}
        }

        # Gather information collected
        # the structure of dtr_data:
        # Level 0: 'breakdown'      | 'throughput'
        # Level 1: data dictionary  | computed throughput (float)
        # Level 3: same as dictionaries processed in fill_data
        for x in x_axis:
            for datum in throughput_metadata[model]['dtr'][x]:
                if datum['memory_budget'] not in dtr_data['throughput']:
                    dtr_data['throughput'][datum['memory_budget']] = []
                    dtr_data['breakdown'][datum['memory_budget']] = []
                dtr_data['throughput'][datum['memory_budget']].append(datum['throughput'] if not datum['error'] else 0)
                dtr_data['breakdown'][datum['memory_budget']].append(dict(filter(lambda x: x[0] != 'throughput',
                                                                    datum.items())) if not datum['error'] else None)

        num_budget = len(dtr_data['throughput'].keys())
        plt.xticks(ind + width * (num_budget / 2), map(str, x_axis))

        for (i, (budget, throughput)) in enumerate(sorted(dtr_data['throughput'].items(), key=lambda x: -x[0])):
            plt.bar(ind + width * (i + 1), throughput, width, label=f'{round(budget * 1e-9, 1)} GiB')

        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')

        # Plot runtime profiling breakdown
        filename = prepare_out_file(output_dir, f'time-breakdown-{model}.png')
        plt.clf()
        plt.title(f'Runtime Breakdown of {NAME_DICT.get(model, model)}')
        plt.xlabel('Batch Size')
        plt.ylabel('Time / Batch (ms)')
        x_ticks_loc = {
            ind[i] + width * (num_budget / 2) : '\n\n' + str(x_axis[i]) for i in range(num_batch_size)
        }
        plt.grid(True, axis='y')
        for (i, (budget, datum)) in enumerate(sorted(dtr_data['breakdown'].items(), key=lambda x: -x[0])):
            locs = ind + width * (i + 1)
            for loc in locs:
                x_tick = f'{round(budget * 1e-9, 1)}\nGiB'
                if loc in x_ticks_loc.keys():
                    x_tick += f'\n{x_ticks_loc[loc]}'
                x_ticks_loc[loc] = x_tick

            if datum is None:
                continue
            gathered_data = {
                key : [] for key in (timed_keys + ['cpu_time'])
            }
            gathered_data['dispatch_overhead'] = []
            for e in datum:
                time_acc = 0
                for key in gathered_data.keys():
                    if key != 'dispatch_overhead':
                        if e is None:
                            gathered_data[key].append(0)
                        else:
                            gathered_data[key].append(e[key])
                        if key != 'cpu_time' and e is not None:
                            time_acc += e[key]
                if e is not None:
                    gathered_data['dispatch_overhead'].append(gathered_data['cpu_time'][-1] - time_acc)
                else:
                    gathered_data['dispatch_overhead'].append(0)

            height_acc = np.zeros(len(datum))
            for key in timed_keys:# + ['dispatch_overhead']:
                if i == 0:
                    plt.bar(ind + width * (i + 1), gathered_data[key],
                            width=width,
                            label=breakdown_namedict[key],
                            color=breakdown_color_scheme[key],
                            bottom=height_acc)
                else:
                    plt.bar(ind + width * (i + 1), gathered_data[key],
                            width=width,
                            color=breakdown_color_scheme[key],
                            bottom=height_acc)

                height_acc += gathered_data[key]
        xticks_data = list(sorted(x_ticks_loc.items(), key=lambda x: -x[0]))
        ticks = list(map(lambda x: x[0], xticks_data))
        labels = list(map(lambda x: x[1], xticks_data))
        plt.xticks(ticks, labels)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')

    try:
        for model in throughput_metadata.keys():
            plot_model(model)
    except Exception as e:
        return False, render_exception(e)

    return True, 'success'

def flatten(xs):
    result = []
    for e in xs:
        if isinstance(e, list) or isinstance(e, tuple) or isinstance(e, np.ndarray):
            result = result + flatten(e)
        else:
            result.append(e)
    return result

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

        success, msg = traverse_field(metadata, 'ratio',
                lambda model, batch_size, dtr_dict, baseline_dict, output_dir:\
                        render_time_comparison(model, batch_size, 'ratio',
                                                dtr_dict[batch_size]['ratio'],
                                                baseline_dict.get(batch_size, {}),
                                                output_dir), output_dir)

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
        # plt.savefig(filename, bbox_inches = 'tight')

        plt.clf()
        plt.rcParams["font.size"] = 30

        figure, axs = plt.subplots(2, 4, figsize=(20, 8))
        # figure.set_size_inches(24, 12)
        axs = reversed(flatten(axs))

        success, msg = traverse_field(metadata, 'fixed',
                lambda model, batch_size, dtr_dict, baseline_dict, output_dir:\
                        render_time_comparison(model, batch_size, 'fixed',
                                                dtr_dict[batch_size]['fixed'],
                                                baseline_dict.get(batch_size, {}), output_dir, plt_ax=next(axs)),
                                                output_dir)

        filename = prepare_out_file(output_dir, 'combined-breakdown-comparison.png')
        # figure.tight_layout()
        # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # plt.xlabel('Memory Budget (GiB)')
        # plt.ylabel("Time (ms)")
        figure.text(0.5, 0.02, r'\textbf{\Huge Memory Budget (GiB)}', ha='center')
        figure.text(0.09, 0.5, r'\textbf{\Huge Time (ms) / Batch}', ha='center', va='center', rotation='vertical')
        plt.legend(
            bbox_to_anchor=(0.17,0.075),
            loc='upper left',
            bbox_transform=fig.transFigure,
            ncol=6,
            borderaxespad=0,
            prop={'size': 15}
        )
       # figure.tight_layout()
        # plt.tight_layout()
        # plt.tight_layout(h_pad=0.3)
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.4)

        if not success:
            return (False, msg)

        success, msg = render_throughput_breakdown(metadata, output_dir)
        if not success:
            return False, msg
        return (True, 'success')
    except Exception as e:
        raise e
        return (False, 'Exception encountered while rendering graphs: {}'.format(render_exception(e)))

def main(data_dir, config_dir, output_dir):
    try:
        config, msg = validate_trials_config(config_dir)
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
