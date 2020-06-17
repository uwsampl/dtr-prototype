from simrd.heuristic import *
from simrd.heuristic.ablation import *

PARETO_MOD = 'eval/pareto'
PAPER_PARETO_HEURISTICS = [
  DTR(), DTREqClass(), DTRLocal(), LRU(), LargestStorage(), RandomStorage()
]
PAPER_ABLATION_HEURISTICS = [
      AbESizeStale(),     AbESize(),     AbEStale(),      AbE(),
     AbEqSizeStale(),    AbEqSize(),    AbEqStale(),     AbEq(),
  AbLocalSizeStale(), AbLocalSize(), AbLocalStale(),  AbLocal(),
       AbSizeStale(),      AbSize(),      AbStale(), AbRandom()
]
PAPER_ACCESSES_HEURISTICS = [
  DTR(), DTREqClass(), DTRLocal()
]

from attr import attrs, attrib
@attrs
class LinePlotSettings:
  marker = attrib()
  color = attrib()
  linestyle = attrib()
  label = attrib()

  @staticmethod
  def from_heuristic(heuristic : Heuristic):
    return LinePlotSettings(heuristic.MARKER, heuristic.COLOR, heuristic.LINESTYLE, str(heuristic))
