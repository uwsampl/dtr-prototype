import os

_OUTPUT_DIR = 'data'

def ensure_output_path(mod):
  p = _OUTPUT_DIR + '/' + mod
  if not os.path.exists(p):
    os.makedirs(p)

def get_output_path(mod, file):
  return _OUTPUT_DIR + '/' + mod + '/' + file
