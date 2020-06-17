class EqClassNode:
  def __init__(self, value, parent=None, tel=None):
    self._value = value
    self._parent = parent
    self.tel = tel

  def is_root(self):
    return self._parent == None

  @staticmethod
  def find_root(ecn : 'EqClassNode'):
    if ecn.tel:
      ecn.tel.summary['heuristic_access_count'] += 1
    if ecn.is_root():
      return ecn
    else:
      ecn._parent = EqClassNode.find_root(ecn._parent)
      return ecn._parent

  @staticmethod
  def get_value(ecn : 'EqClassNode'):
    return EqClassNode.find_root(ecn)._value

  @staticmethod
  def set_value(ecn : 'EqClassNode', value):
    EqClassNode.find_root(ecn)._value = value

  @staticmethod
  def merge(merge_f, lhs : 'EqClassNode', rhs : 'EqClassNode'):
    l = EqClassNode.find_root(lhs)
    r = EqClassNode.find_root(rhs)
    if l == r:
      return l
    l._parent = r
    r._value = merge_f(l._value, r._value)
    return r

class CheckpointInfo:
  def __init__(self, compute):
    self.compute = compute

  @staticmethod
  def merge_f(l : 'CheckpointInfo', r : 'CheckpointInfo'):
    return CheckpointInfo(
      l.compute + r.compute
    )
