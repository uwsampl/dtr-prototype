import math

BOUNDS = {}

def register_bound(cls):
  cls.ID = len(BOUNDS)
  b_str = str(cls())
  assert b_str not in BOUNDS, \
    'cannot have the same description as an existing bound: {}'.format(cls)
  BOUNDS[b_str] = cls
  return cls

class MemoryBound:
  def __call__(self):
    raise NotImplementedError

@register_bound
class TQBound(MemoryBound):
  def __call__(self, n):
    return math.ceil(2 * math.sqrt(n))

  def __str__(self):
    return r'$2\sqrt{n}$'

@register_bound
class SqrtBound(MemoryBound):
  def __call__(self, n):
    return math.ceil(math.sqrt(n))

  def __str__(self):
    return r'$\sqrt{n}$'

@register_bound
class Log2Bound(MemoryBound):
  def __call__(self, n):
    return math.ceil(math.log2(n))

  def __str__(self):
    return r'$\log_2(n)$'

@register_bound
class LinearBound(MemoryBound):
  def __call__(self, n):
    return n

  def __str__(self):
    return r'$n$'

@register_bound
class HalfLinearBound(MemoryBound):
  def __call__(self, n):
    return math.ceil(n / 2)

  def __str__(self):
    return r'$n / 2$'

@register_bound
class SqrtBound4(MemoryBound):
  def __call__(self, n):
    return math.ceil(4 * math.sqrt(n))

  def __str__(self):
    return r'$4\sqrt{n}$'

@register_bound
class SqrtBound8(MemoryBound):
  def __call__(self, n):
    return math.ceil(8 * math.sqrt(n))

  def __str__(self):
    return r'$8\sqrt{n}$'

@register_bound
class SqrtBound6(MemoryBound):
  def __call__(self, n):
    return math.ceil(6 * math.sqrt(n))

  def __str__(self):
    return r'$6\sqrt{n}$'

@register_bound
class Log2Bound3(MemoryBound):
  def __call__(self, n):
    return math.ceil(3 * math.log2(n))

  def __str__(self):
    return r'$3\log_2(n)$'
