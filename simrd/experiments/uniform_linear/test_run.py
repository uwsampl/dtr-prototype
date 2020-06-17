from simrd.heuristic import *
from simrd.runtime import *
from simrd.telemetry import Telemetry
from experiments.bounds import TQBound

from experiments.uniform_linear.run import run

if __name__ == '__main__':
  n = 1000
  B = TQBound()(n)
  r = RuntimeV2EagerOptimized
  rt_kwargs = {'trace': True, 'stats': True}

  print('running ({})...'.format(r.ID))
  import time
  t = time.time()
  rt = run(n, B, DTR(), r, releases=True, rt_kwargs=rt_kwargs)
  print('  done, took {} seconds.'.format(time.time() - t))
  print(rt.telemetry.summary)
  import pandas as pd
  df = pd.DataFrame(rt.telemetry.tensor.values(), columns=Telemetry.TENSOR_STATS)
  df2 = pd.DataFrame(rt.telemetry.storage.values(), columns=Telemetry.STORAGE_STATS)
  df3 = pd.DataFrame(rt.telemetry.operator.values(), columns=Telemetry.OPERATOR_STATS)
  import pdb; pdb.set_trace()
