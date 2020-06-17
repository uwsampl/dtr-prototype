"""
Reads experiment summaries and posts them to Slack.
"""
import argparse
import json
import os
import textwrap

from common import invoke_main, read_config, write_status, read_json, validate_json
from dashboard_info import DashboardInfo
from slack_util import (generate_ping_list,
                        build_field, build_attachment, build_message,
                        post_message)

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


def main(config_dir, home_dir, output_dir):
    config = read_config(config_dir)
    if 'webhook_url' not in config:
        write_status(output_dir, False, 'No webhook URL given')
        return 1

    webhook = config['webhook_url']
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

    success, report = post_message(
        webhook,
        build_message(
            text='*Dashboard Results*{}'.format(
                '\n' + description if description != '' else ''),
            attachments=attachments))
    write_status(output_dir, success, report)


if __name__ == '__main__':
    invoke_main(main, 'config_dir', 'home_dir', 'output_dir')
