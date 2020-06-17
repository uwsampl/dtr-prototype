import json, glob

from experiments.util import *

from .definitions import *

def get_result_heuristic_str(result_file):
  results = None
  with open(result_file, 'r') as rf:
    results = json.loads(rf.read())
  assert results is not None
  return results['config']['heuristic']

def get_top_k_bases(k):
  base_dir = get_output_path(PARETO_MOD, '')
  bases_all = [os.path.basename(d) for d in glob.glob(base_dir + '*') if os.path.isdir(d)]
  bases_all = sorted(bases_all, reverse=True)
  return bases_all[:k]
