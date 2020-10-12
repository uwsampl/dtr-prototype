"""
Just prints the location of the configured setup directory
so it can be used by bash commands.

This is a total nasty hack to deal with the fact that
there is no global dependency management in the dashboard.
"""
import os

from common import invoke_main, read_config

def sanitize_path(path):
    return os.path.abspath(os.path.expanduser(path))

def main(home_dir):
    global_conf = read_config(sanitize_path(home_dir))
    print(sanitize_path(global_conf['setup_dir']))

if __name__ == '__main__':
    invoke_main(main, 'home_dir')
