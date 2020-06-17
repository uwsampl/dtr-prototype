import pandas as pd
import matplotlib.pyplot as plt

from graphviz import Digraph
from simrd.telemetry import Telemetry

TRACE_STATS = [
  'time', 'pinned_memory', 'locked_memory', 'evictable_memory', 'total_memory', 'memory_pressure'
]

class State:
  def __init__(self, telemetry : Telemetry):
    self.material  = set()
    self.evicted   = set()
    self.locked    = set()
    self.pending   = set()
    self.pinned    = set()
    self.banished  = set()
    self.timesteps = sorted(telemetry.trace.timesteps)
    self.time_idx  = -1
    self.time      = 0
    self.pressure  = 0

    self.telemetry = telemetry

    # initialize to be all evicted and uncomputed
    self.evicted.update(list(self.telemetry.tensor.keys()))
    self.uncomputed = self.evicted.copy()

    # group tensors by their parent opcalls
    self.call_groups = {}
    for tid in self.evicted:
      op_id = self.telemetry.get('tensor', tid, 'op_id')
      if op_id not in self.call_groups:
        self.call_groups[op_id] = []
      self.call_groups[op_id].append(tid)

    # group tensors by their underlying storages
    self.storage_groups = {}
    for tid in self.evicted:
      storage_id = self.telemetry.get('tensor', tid, 'storage_id')
      if storage_id not in self.storage_groups:
        self.storage_groups[storage_id] = []
      self.storage_groups[storage_id].append(tid)

  def _step(self):
    if self.time_idx + 1 >= len(self.timesteps):
      return False

    self.time_idx += 1
    self.time = self.timesteps[self.time_idx]

    # A tensor cannot be evicted *and then* computed on the same timestep,
    # only computed then evicted. Thus, we can set computed = computed \ evicted.

    # evicted /\ material = empty
    # pending subset evicted
    # banished subset evicted
    # locked subset material
    # pinned subset material
    # locked /\ pinned = empty

    # add new material tensors
    compute = self.telemetry.trace.compute.get(self.time, [])
    self.material.update(compute)
    self.evicted.difference_update(compute)
    self.pending.difference_update(compute)
    self.uncomputed.difference_update(compute)

    # add new evicted and banished tensors
    for sid in self.telemetry.trace.evict.get(self.time, []):
      self.evicted.update(self.storage_groups[sid])
    for sid in self.telemetry.trace.banish.get(self.time, []):
      self.evicted.update(self.storage_groups[sid])
      self.banished.update(self.storage_groups[sid])

    # add new pending tensors
    self.pending.update(self.telemetry.trace.pending.get(self.time, []))

    # update pinned, locked; a lock can only be locked after being unlocked at
    # the same timestep, *assuming computations take nonzero time*
    for sid in self.telemetry.trace.unlock.get(self.time, []):
      self.locked.difference_update(self.storage_groups[sid])
    for sid in self.telemetry.trace.lock.get(self.time, []):
      self.locked.update(self.storage_groups[sid])

    for sid in self.telemetry.trace.pin.get(self.time, []):
      self.pinned.update(self.storage_groups[sid])

    self.material.difference_update(self.evicted)
    self.locked.difference_update(self.evicted)
    self.pinned.difference_update(self.evicted)

    # Track pinned and locked separately
    for tid in self.pinned:
      if tid in self.storage_groups:
        self.locked.difference_update(self.storage_groups[tid])

    # Mark any material tensors belonging to a pinned group as pinned
    for tid in self.material:
      sid = self.telemetry.get('tensor', tid, 'storage_id')
      if sid in self.pinned:
        self.pinned.add(tid)

    self.pressure = self.telemetry.trace.pressure.get(self.time, 0)

    assert len(self.material.intersection(self.evicted)) == 0
    assert len(self.banished.intersection(self.evicted)) == len(self.banished)
    assert len(self.locked.intersection(self.pinned)) == 0
    assert len(self.locked.intersection(self.evicted)) == 0
    assert len(self.locked.intersection(self.material)) == len(self.locked)
    assert len(self.material) + len(self.evicted) == len(self.telemetry.tensor.keys())

    return True

  def step(self, steps=1):
    for i in range(steps):
      if not self._step():
        return False
    return True

  def tensor_name(self, tid):
    op_id = self.telemetry.get('tensor', tid, 'op_id')
    op_name = self.telemetry.get('operator', op_id, 'name')
    if op_name == None:
      op_name = self.telemetry.get('tensor', tid, 'name')
    if self.telemetry.get('operator', op_id, 'outputs') > 1:
      index = self.telemetry.get('tensor', tid, 'index')
      name = '{}.{}'.format(op_name, index)
    else:
      name = op_name
    return name

  def render_dot(self, filename=None, **kwargs):
    """
    Returns the dot graph (of the current state) as a string, and optionally
    outputs a rendered image to the specified filename (without file extension;
    the extension and format can be changed by setting format='png', etc.
    although pdf seems to be the fastest).
    """

    g = Digraph(engine='dot')
    g.attr('node', shape='box', style='filled')
    g.attr(compound='true', rankdir='LR')

    for call_id in self.call_groups:
      with g.subgraph(name='cluster_{}'.format(call_id)) as gg:
        gg.attr(style='filled', color='lightgrey')
        for tid in self.call_groups[call_id]:
          if tid in self.material:
            args = {'color': 'white'}
            if tid in self.pinned:
              args['color'] = 'orange'
            elif tid in self.locked:
              args['color'] = 'pink'
          elif tid in self.evicted:
            args = {'color': 'gray'}
            if tid in self.banished:
              args['color'] = 'red'
            elif tid in self.uncomputed:
              args['color'] = 'white'
              args['fillcolor'] = 'black'
              args['fontcolor'] = 'white'
              args['style'] = 'dashed, filled'
            if tid in self.pending:
              args['fillcolor'] = 'darkgreen'
              args['fontcolor'] = 'white'
          gg.node(str(tid), self.tensor_name(tid), **args)

    # edges, only draw one per op output cluster
    for p in self.telemetry.adj_list:
      c_op_ids = set()
      for c in self.telemetry.adj_list[p]:
        c_op_id = self.telemetry.get('tensor', c, 'op_id')
        if c_op_id not in c_op_ids:
          c_op_ids.add(c_op_id)
        else:
          continue
        c_cluster = 'cluster_{}'.format(c_op_id)
        args = {}
        if self.telemetry.get('tensor', c, 'is_alias'):
          args['style'] = 'dashed'
        g.edge(str(p), str(c), lhead=c_cluster, **args)

    if filename is not None:
      g.render(filename, cleanup=True, **kwargs)

    return g.source

def analyze_trace(tel : Telemetry):
  s = State(tel)
  data = []
  while s.step():
    pinned_mem       = 0
    locked_mem       = 0
    evictable_mem    = 0
    total_mem        = 0
    for tid in s.pinned:
      pinned_mem += s.telemetry.get('tensor', tid, 'size')
    for tid in s.locked:
      locked_mem += s.telemetry.get('tensor', tid, 'size')
    for tid in s.material.difference(s.pinned).difference(s.locked):
      evictable_mem += s.telemetry.get('tensor', tid, 'size')
    total_mem = pinned_mem + locked_mem + evictable_mem
    mem_pressure = total_mem + s.pressure
    data.append([s.time, pinned_mem, locked_mem, evictable_mem, total_mem, mem_pressure])
  return pd.DataFrame(data, columns=TRACE_STATS)

def analyze_max_pinned(tel : Telemetry, filename, render_graph=False):
  s = State(tel)
  max_pinned_memory = -1
  max_pinned_step = -1
  max_pinned_time = -1
  while s.step():
    pinned_mem = sum(map(lambda tid: s.telemetry.get('tensor', tid, 'size'), s.pinned))
    if pinned_mem > max_pinned_memory:
      max_pinned_memory = pinned_mem
      max_pinned_step = s.time_idx
      max_pinned_time = s.time
  print('maximum amount of pinned memory: {} at step {} (time = {})'.format(
    max_pinned_memory, max_pinned_step, max_pinned_time
  ))
  # now graph
  s = State(tel)
  while s.step():
    if s.time_idx == max_pinned_step:
      if render_graph:
        s.render_dot(filename)
      pinned_data = []
      for tid in s.pinned:
        pinned_data.append(s.telemetry.tensor[tid])
      stats = pd.DataFrame(pinned_data, columns=Telemetry.TENSOR_STATS)
  return stats

def analyze_max_locked(tel : Telemetry, filename, render_graph=False):
  s = State(tel)
  max_locked_memory = -1
  max_locked_step = -1
  max_locked_time = -1
  while s.step():
    locked_mem = sum(map(lambda tid: s.telemetry.get('tensor', tid, 'size'), s.locked))
    if locked_mem > max_locked_memory:
      max_locked_memory = locked_mem
      max_locked_step = s.time_idx
      max_locked_time = s.time
  print('maximum amount of locked memory: {} at step {} (time = {})'.format(
    max_locked_memory, max_locked_step, max_locked_time
  ))
  # now graph
  s = State(tel)
  while s.step():
    if s.time_idx == max_locked_step:
      if render_graph:
        s.render_dot(filename)
      pinned_data = []
      for tid in s.pinned:
        pinned_data.append(s.telemetry.tensor[tid])
      stats = pd.DataFrame(pinned_data, columns=Telemetry.TENSOR_STATS)
  return stats

def memory_analysis(tel : Telemetry, stats : pd.DataFrame, start=0, end=None):
  stats = stats[start:end]
  plt.plot(stats.index, stats['evictable_memory'], alpha=0.8, color='green', label='evictable memory')
  plt.plot(stats.index, stats['memory_pressure'], alpha=0.8, color='red', label='memory pressure')
  plt.plot(stats.index, stats['pinned_memory'], alpha=0.8, color='orange', label='pinned memory')
  plt.plot(stats.index, stats['locked_memory'], alpha=0.9, color='pink', label='locked memory')
  plt.plot(stats.index, stats['total_memory'], alpha=0.5, color='blue', label='total memory usage')
  plt.axhline(y=tel.summary['max_memory'], linestyle='dotted', label='peak/max memory')
  plt.xlabel('Operators Computed (in order)')
  plt.ylabel('Memory (Bytes)')
  plt.legend()
