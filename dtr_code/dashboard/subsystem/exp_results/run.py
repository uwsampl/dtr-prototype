"""
Reads experiment summaries and posts them to Slack.
"""
import argparse
import json
import os
import textwrap

from common import invoke_main, read_config, write_status, read_json, validate_json
from dashboard_info import DashboardInfo
from slack import WebClient
from slack_util import (generate_ping_list,
                        build_field, build_attachment, build_message,
                        post_message, new_client, upload_image)

def attach_duration(message, duration=None):
    if duration is None:
        return message
    return '_Duration: {}_\n\n{}'.format(duration, message)


def failed_experiment_field(exp, stage_statuses, stage, duration=None, notify=None):
    message = 'Failed at stage {}:\n{}'.format(
        stage,
        textwrap.shorten(stage_statuses[stage]['message'], width=280))

    if duration is not None:
        message += '\nTime to failure: {}'.format(duration)

    if notify is not None:
        message += '\nATTN: {}'.format(generate_ping_list(notify))

    return build_field(title=exp, value=message)

def send_graphs(config, info, client, output_dir):
    img_dict = dict()
    for (curr_dir, _, files) in os.walk(info.exp_graphs):
        for filename in files:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                if curr_dir not in img_dict:
                    img_dict[curr_dir] = []
                img_dict[curr_dir].append(filename)
    for (dir_name, files) in img_dict.items():
        exp_name = dir_name.split(os.path.sep)[-1] if os.path.sep in dir_name else dir_name
        success, resp, msg = post_message(client, config['channel_id'], build_message(text=f'Graphs of {exp_name}'))
        if not success:
            return (False, msg)
        channel_thread_ts = list(map(lambda resp: (resp.data['channel'], resp.data['ts']), resp))
        for filename in files:
            file_path = f'{dir_name}/{filename}'
            for channel, thread_ts in channel_thread_ts:
                success, _, msg = upload_image(client, channel, file_path, filename, thread_ts=thread_ts)
                if not success:
                    return (False, msg)
    return True, 'success'

def main(config_dir, home_dir, output_dir):
    config = read_config(config_dir)
    if 'channel_id' not in config:
        write_status(output_dir, False, 'No channel token given')
        return 1
    
    success, msg, client = new_client(config)
    info = DashboardInfo(home_dir)

    if not success:
        write_status(output_dir, False, msg)
        return 1

    slack_channel = config['channel_id']
    description = ''
    if 'description' in config:
        description = config['description']

    info = DashboardInfo(home_dir)

    inactive_experiments = []     # list of titles
    failed_experiments = []       # list of slack fields
    successful_experiments = []   # list of slack fields
    failed_graphs = []            # list of titles

    for exp_name in info.all_present_experiments():
        stage_statuses = info.exp_stage_statuses(exp_name)
        if not stage_statuses['precheck']['success']:
            failed_experiments.append(
                failed_experiment_field(exp_name, stage_statuses, 'precheck'))
            continue

        exp_conf = info.read_exp_config(exp_name)
        exp_status = info.exp_status_dir(exp_name)
        run_status = validate_json(exp_status, 'time_delta', filename='run.json')

        exp_title = exp_name if 'title' not in exp_conf else exp_conf['title']
        notify = exp_conf['notify']
        if not exp_conf['active']:
            inactive_experiments.append(exp_title)
            continue

        failure = False
        for stage in ['setup', 'run', 'analysis', 'summary']:
            if stage not in stage_statuses:
                # setup is the only stage that's optional
                assert stage == 'setup'
                continue
            if not stage_statuses[stage]['success']:
                failed_experiments.append(
                    failed_experiment_field(exp_title, stage_statuses,
                                            stage,
                                            duration=run_status.get('time_delta'),
                                            notify=notify))
                failure = True
                break

        if failure:
            continue

        # failure to visualize is not as big a deal as failing to
        # run or analyze the experiment, so we only report it but
        # don't fail to report the summary
        if not stage_statuses['visualization']['success']:
            failed_graphs.append(exp_title)

        summary = info.read_exp_summary(exp_name)
        successful_experiments.append(
            build_field(summary['title'],
                        attach_duration(summary['value'],
                                        run_status.get('time_delta'))))

    # produce messages
    attachments = []
    if successful_experiments:
        attachments.append(
            build_attachment(
                title='Successful benchmarks',
                fields=successful_experiments))
    if failed_experiments:
        attachments.append(
            build_attachment(
                color='#fa0000',
                title='Failed benchmarks',
                fields=failed_experiments))
    if inactive_experiments:
        attachments.append(
            build_attachment(
                color='#616161',
                title='Inactive benchmarks',
                text=', '.join(inactive_experiments)))
    if failed_graphs:
        attachments.append(
            build_attachment(
                color='#fa0000',
                title='Failed to Visualize',
                text=', '.join(failed_graphs)))

    success, _, report = post_message(
        client,
        slack_channel,
        build_message(
            text='*Dashboard Results*{}'.format(
                '\n' + description if description != '' else ''),
            attachments=attachments))
    if config.get('report_images', False):
        success, msg = send_graphs(config, info, client, output_dir)
        if not success:
            write_status(output_dir, False, msg)
            return 1

    write_status(output_dir, success, report)

if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')
