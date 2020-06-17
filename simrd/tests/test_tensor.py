from simrd.tensor import *

def test_operator():
  op = Operator(100, (10, 20, 0), (-1, -1, 0), name='f')
  assert op.compute == 100
  assert op.sizes == (10, 20, 0)
  assert op.total_size == 30
  assert op.name == 'f'
  assert op.outputs == 3

def test_simple_tensors():
  op = Operator(10, (5,), (-1,), name='f')
  (t,) = Tensor.from_op([], op, 0, (0,), ('t',))

  assert len(t.parents) == 0 and len(t.siblings) == 0 and len(t.children) == 0
  assert t.index == 0
  assert t.op.name == 'f'
  assert t.id == 0
  assert t.name == 't'
  assert not t.meta['is_alias']
  assert not t.defined
  assert t.meta['ref_ext'] == 1

  assert t.storage.root_id == 0
  assert t.storage.size == 5
  assert t.storage.ref_ext == 1
  assert t.storage.ref_int == 0
  assert len(t.storage.tensors) == 1
  assert t.storage.tensors[0].id == 0

  op2 = Operator(20, (4,), (-1,), name='f2')
  (x1,) = Tensor.from_op([t], op2, 0, (1,))

  assert len(t.children) == 1
  assert t.children[0] == x1

  assert len(x1.parents) == 1 and len(x1.siblings) == 0 and len(x1.children) == 0
  assert x1.id == 1 and x1.name == 'x1'
  assert x1.op.name == 'f2'
  assert not x1.meta['is_alias'] and not x1.defined
  assert x1.storage.size == 4

def test_tuple_tensors():
  op = Operator(10, (5,10), (-1,-1), name='f_tuple')
  (t, u) = Tensor.from_op([], op, 0, (0,1))

  assert t.name == 'x0' and u.name == 'x1'
  assert len(t.siblings) == 1 and len(u.siblings) == 1
  assert t.siblings[0] == u and u.siblings[0] == t
  assert t.index == 0 and u.index == 1

  (t2, u2) = Tensor.from_op([t, u], op, 1, (2, 3))

  assert t2.name == 'x2' and u2.name == 'x3'
  assert len(t2.parents) == 2 and len(u2.parents) == 2
  assert len(t.children) == 2 and len(u.children) == 2
  assert t2 in t.children and u2 in t.children and t2 in u.children and u2 in u.children

def test_simple_alias():
  op1 = Operator(10, (5,), (-1,), name='f1')
  op_alias = Operator(1, (0,), (0,), name='alias')
  (t,) = Tensor.from_op([], op1, 0, (0,))

  (t_alias,) = Tensor.from_op([t], op_alias, 1, (1,))

  assert t_alias.storage == t.storage
  assert t.storage.ref_ext == 2
  assert t_alias.meta['is_alias']

  (t_alias_2,) = Tensor.from_op([t_alias], op_alias, 2, (2,))

  assert t_alias_2.storage == t.storage
  assert t.storage.ref_ext == 3
  assert t_alias.meta['is_alias']
