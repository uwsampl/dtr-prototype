"""
Python utilities for posting to Slack
"""
import datetime
import json
import logging
import os
import requests
import sys

from slack import WebClient
from slack.errors import SlackApiError
from common import render_exception

def generate_ping_list(user_ids):
    return ', '.join(['<@{}>'.format(user_id) for user_id in user_ids])

def new_client(config):
    try:
        # Please keep token locally. Do not upload config file(s) to the GitHub Repo.
        return True, 'success', WebClient(token=config['slack_client_token'])
    except Exception as e:
        return False, f'Exception encountered: {render_exception(e)}', None

def build_field(title='', value='', short=False):
    ret = {'value': value, 'short': short}
    if title != '':
        ret['title'] = title
    return ret


def build_attachment(title='', fields=None, pretext='', text='', color='#000000'):
    ret = {'color': color}

    if title != '':
        ret['title'] = title
    if fields is not None:
        ret['fields'] = fields
    if pretext != '':
        ret['pretext'] = pretext
    if text != '':
        ret['text'] = text

    return ret


def build_message(text='', pretext='', attachments=None):
    ret = {'text': text}
    if 'pretext' != '':
        ret['pretext'] = pretext
    if attachments is not None:
        ret['attachments'] = attachments

    return ret


def post_message(client, channel, message, **kargs):
    """
    Attempts posting the given message object to the
    Slack channel.
    Returns whether it was successful and a message.
    """
    try:
        resp = []
        if isinstance(channel, list):
            for ch in channel:
                resp.append(client.chat_postMessage(channel=ch, text=message.get('text', '*No Message Content*'), attachments=message.get('attachments', ''), **kargs))
        else:
            resp.append(client.chat_postMessage(channel=channel, text=message.get('text', '*No Message Content*'), attachments=message.get('attachments', ''), **kargs))
        return (True, resp, 'success')
    except Exception as e:
        return (False, None, 'Encountered exception:\n' + render_exception(e))

def upload_image(client, channels, file_path, description, **kargs):
    """
    Attempts to upload an image to a channel
    """
    try:
        if isinstance(channels, list):
            channels = ','.join(channels)
        resp = client.files_upload(channels=channels, file=file_path, title=description, **kargs)
        return (True, resp, 'success')
    except Exception as e:
        return (False, None, 'Encountered exception:\n' + render_exception(e))
