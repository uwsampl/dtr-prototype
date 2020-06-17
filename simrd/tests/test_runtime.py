from simrd.runtime import *
from simrd.heuristic import DTRUnopt

OP1 = Operator(2, (1,), (-1,), name='op1')
OP2 = Operator(2, (1,1), (-1,-1), name='op2')
OPA1 = Operator(1, (0,), (0,), name='opa1')

def test_simple_V1():
  rt = RuntimeV1(math.inf, DTRUnopt)
  (x,) = rt.compute([], OP1)

  assert x.storage.material and x.defined
  assert x.storage.ref_ext == 1 and x.storage.ref_int == 0
  assert x.storage.root_id == x.id

  (y,) = rt.compute([x], OP1)

  assert x.storage.material and x.defined
  assert y.storage.material and y.defined
  assert x.storage.ref_ext == 1 and x.storage.ref_int == 0
  assert y.storage.ref_ext == 1 and y.storage.ref_int == 0

  assert x in y.parents
  assert y.storage.root_id == y.id

def test_simple_V2():
  rt = RuntimeV2(math.inf, DTRUnopt)
  (x,) = rt.compute([], OP1)

  assert x.storage.material and x.defined
  assert x.storage.ref_ext == 1 and x.storage.ref_int == 0
  assert x.storage.root_id == x.id

  (y,) = rt.compute([x], OP1)

  assert x.storage.material and x.defined
  assert y.storage.material and y.defined
  assert x.storage.ref_ext == 1 and x.storage.ref_int == 0
  assert y.storage.ref_ext == 1 and y.storage.ref_int == 0

  assert x in y.parents
  assert y.storage.root_id == y.id
