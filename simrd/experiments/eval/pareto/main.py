from experiments.eval.pareto.run import *
from experiments.eval.pareto.plot import *

if __name__ == '__main__':
  run_ablation_paper()
  plot_ablation_paper()

  run_pareto_paper()
  plot_pareto_paper()
  plot_accesses_paper()

  run_banishing_paper()
  plot_banishing_paper()
