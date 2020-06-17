"""
Common Python utilities for interacting with the dashboard infra.
"""
import argparse
import datetime
import json
import logging
import os
import sys


def print_log(msg, dec_char='*'):
    padding = max(list(map(len, str(msg).split('\n'))))
    decorate = dec_char * (padding + 4)
    print(f'{decorate}\n{msg}\n{decorate}')

def validate_json(dirname, *fields, filename='status.json'):
    if not check_file_exists(dirname, filename):
        return {'success': False, 'message': 'No {} in {}'.format(filename, dirname)}
    fp = read_json(dirname, filename)
    for required_field in fields:
        if required_field not in fp:
            return {'success': False,
                    'message': '{} in {} has no \'{}\' field'.format(filename, dirname, required_field)}
    return fp

def check_file_exists(dirname, filename):
    dirname = os.path.expanduser(dirname)
    full_name = os.path.join(dirname, filename)
    return os.path.isfile(full_name)


def idemp_mkdir(dirname):
    '''Creates a directory in an idempotent fashion.'''
    dirname = os.path.expanduser(dirname)
    os.makedirs(dirname, exist_ok=True)


def prepare_out_file(dirname, filename):
    dirname = os.path.expanduser(dirname)
    full_name = os.path.join(dirname, filename)
    if not check_file_exists(dirname, filename):
        os.makedirs(os.path.dirname(full_name), exist_ok=True)
    return full_name


def read_json(dirname, filename):
    dirname = os.path.expanduser(dirname)
    with open(os.path.join(dirname, filename)) as json_file:
        data = json.load(json_file)
        return data


def write_json(dirname, filename, obj):
    filename = prepare_out_file(dirname, filename)
    with open(filename, 'w') as outfile:
        json.dump(obj, outfile)


def read_config(dirname):
    return read_json(dirname, 'config.json')


def write_status(output_dir, success, message):
    write_json(output_dir, 'status.json', {
        'success': success,
        'message': message
    })


def write_summary(output_dir, title, value):
    write_json(output_dir, 'summary.json', {
        'title': title,
        'value': value
    })


def get_timestamp():
    time = datetime.datetime.now()
    return time.strftime('%m-%d-%Y-%H%M')


def parse_timestamp(data):
    return datetime.datetime.strptime(data['timestamp'], '%m-%d-%Y-%H%M')


def time_difference(entry1, entry2):
    '''
    Returns a datetime object corresponding to the difference in
    timestamps between two data entries.
    (Entry 1 time - entry 2 time)
    '''
    return parse_timestamp(entry1) - parse_timestamp(entry2)


def sort_data(data_dir):
    '''Sorts all data files in the given directory by timestamp.'''
    data_dir = os.path.expanduser(data_dir)
    all_data = []
    for _, _, files in os.walk(data_dir):
        for name in files:
            data = read_json(data_dir, name)
            all_data.append(data)
    return sorted(all_data, key=parse_timestamp)


def gather_stats(sorted_data, fields):
    '''
    Expects input in the form of a list of data objects with timestamp
    fields (like those returned by sort_data).
    For each entry, this looks up entry[field[0]][field[1]]...
    for all entries that have all the fields, skipping those that
    don't. Returns a pair (list of entry values,
    list of corresponding entry timestamps)
    '''
    stats = []
    times = []
    for entry in sorted_data:
        stat = entry
        not_present = False
        for field in fields:
            if field not in stat:
                not_present = True
                break
            stat = stat[field]
        if not_present:
            continue
        times.append(parse_timestamp(entry))
        stats.append(stat)
    return (stats, times)


def traverse_fields(entry, ignore_fields=None):
    """
    Returns a list of sets of nested fields (one set per level of nesting)
    of a JSON data entry produced by a benchmark analysis script.
    Ignores the 'detailed' field by default (as old data files will not have detailed summaries).
    Set ignore_fields to a non-None value to avoid the defaults.
    """
    ignore_set = {'timestamp', 'detailed', 
                  'start_time', 'end_time', 'time_delta', 'success',
                  'run_cpu_telemetry', 'run_gpu_telemetry'}
    if ignore_fields is not None:
        ignore_set = set(ignore_fields)

    level_fields = {field for field in entry.keys()
                    if field not in ignore_set}
    values_to_check = [entry[field] for field in level_fields
                       if isinstance(entry[field], dict)]

    tail = []
    max_len = 0
    for value in values_to_check:
        next_fields = traverse_fields(value)
        tail.append(next_fields)
        if len(next_fields) > max_len:
            max_len = len(next_fields)

    # combine all the field lists (union of each level's sets)
    final_tail = []
    for i in range(max_len):
        u = set({})
        final_tail.append(u.union(*[fields_list[i]
                                    for fields_list in tail
                                    if len(fields_list) > i]))

    return [level_fields] + final_tail


def invoke_main(main_func, *arg_names):
    """
    Generates an argument parser for arg_names and calls
    main_func with the arguments it parses. Arguments
    are assumed to be string-typed. The argument names should
    be Python-valid names.
    If main_func returns a value, this function assumes it to
    be a return code. If not, this function will exit with code
    0 after invoking main
    """
    parser = argparse.ArgumentParser()
    for arg_name in arg_names:
        name = arg_name
        parser.add_argument('--{}'.format(name.replace('_', '-')),
                            required=True, type=str)
    args = parser.parse_args()
    ret = main_func(*[getattr(args, name) for name in arg_names])
    if ret is None:
        sys.exit(0)
    sys.exit(ret)


def render_exception(e):
    return logging.Formatter.formatException(e, sys.exc_info())
