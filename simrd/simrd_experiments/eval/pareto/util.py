import json, glob

from ...util import *
from .definitions import *

def get_result_heuristic_str(result_file):
  results = None
  with open(result_file, 'r') as rf:
    results = json.loads(rf.read())
  assert results is not None
  return results['config']['heuristic']

def get_top_k_base_dirs(k, output_dir):
  bases_all = [d for d in glob.glob(output_dir + '/' + '*') if os.path.isdir(d)]
  bases_all = sorted(bases_all, reverse=True)
  return bases_all[:k]
