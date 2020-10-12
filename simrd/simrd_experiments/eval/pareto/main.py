import sys

from simrd_experiments.eval.pareto.run import *
from simrd_experiments.eval.pareto.plot import *
from simrd_experiments.eval.models import MANIFEST

def main():
  sys.setrecursionlimit(1000000000)

  ablation_dir = 'data/ablation'
  pareto_dir = 'data/pareto'
  banishing_dir = 'data/banishing'

  run_models = MANIFEST.values()

  run_pareto_paper(models=run_models, output_dir=pareto_dir)
  run_banishing_paper(models=run_models, output_dir=banishing_dir)
  run_ablation_paper(models=run_models, output_dir=ablation_dir)

  plot_models = MANIFEST.values()
  num_models = len(plot_models)

  plot_pareto_paper(output_dir=pareto_dir, plot_file='data/pareto.pdf', num_models=num_models)
  plot_banishing_paper(output_dir=banishing_dir, plot_file='data/banishing.pdf', num_models=num_models)
  plot_ablation_paper(output_dir=ablation_dir, plot_file='data/ablation_{}.pdf', num_models=num_models)
  plot_accesses_paper(output_dir=pareto_dir, plot_file='data/accesses.pdf', num_models=num_models)

# def main():
#   pareto_base_dirs = run_pareto_paper()
#   banishing_base_dirs = run_banishing_paper()
#   ablation_base_dirs = run_ablation_paper()

#   plot_pareto_paper(base_dirs=pareto_base_dirs, plot_file='data/eval/pareto.png')
#   plot_banishing_paper(base_dirs=banishing_base_dirs, plot_file='data/eval/banishing.png')
#   plot_ablation_paper(base_dirs=ablation_base_dirs, plot_file='data/eval/ablation_{}.png')
#   plot_accesses_paper(base_dirs=pareto_base_dirs, plot_file='data/eval/accesses.png')

if __name__ == '__main__':
  main()
