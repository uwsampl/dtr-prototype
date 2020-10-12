import os
from datetime import datetime

_OUTPUT_DIR = 'data'

def ensure_path(path):
  if not os.path.exists(path):
    os.makedirs(path)

def ensure_output_path(mod):
  p = _OUTPUT_DIR + '/' + mod
  ensure_path(p)

def get_output_dir(mod):
  return _OUTPUT_DIR + '/' + mod

def get_output_path(mod, file):
  return get_output_dir(mod) + '/' + file

def date_string():
  return datetime.now().strftime("%Y%m%d-%H%M%S")
