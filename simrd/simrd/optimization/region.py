class Region:
  """
  Stores a reachable set of evicted storages (respective to some base storage),
  in the specified direction (computation graph normally, reversed graph if
  `reverse == True`). Additionally, stores a set of "frontier" storages that are
  material, representing the edge of the region.

  `Region`s are designed for optimizing heuristics that can be thought of as
  folds over evicted neighborhoods, such as the DTR heuristic (where the fold
  operation is summing costs and maxing last_access).
  """
  def __init__(self, base, reverse : bool):
    self.base = base
    self.interior : Set[int] = set()
    self.frontier : Set[int] = set()
    self.compute = 0
    self.reverse = reverse
  
  def visitor(self, t):
    if self.reverse:
      return t.parents
    return t.children

  def absorb(self, frontier_s, tensor_map, tel : 'Telemetry'):
    """
    Absorbs the corresponding Region for `frontier_s` (i.e., the Region of the
    same direction), by removing `frontier_s` from the frontier for this Region
    and adding it to the interior, while incorporating its interior. Adds the
    frontier of the absorbed Region to this Region's frontier.

    NOTE: This should be called after `frontier_s` is evicted, on its frontiers.

    INVARIANTS: not frontier_s.material
                frontier_s.root_id in self.frontier
    """
    assert not frontier_s.material
    assert frontier_s.root_id in self.frontier

    reg = frontier_s.meta['region_rev'] if self.reverse else frontier_s.meta['region']

    self.frontier.remove(frontier_s.root_id)
    self.frontier.update(reg.frontier)

    # TODO: the difference in the other direction might be smaller, generally
    interior_diff = reg.interior.difference(self.interior)
    interior_diff.add(frontier_s.root_id)

    for sid in interior_diff:
      tel.summary['heuristic_access_count'] += 1
      interior_s = tensor_map[sid].storage
      self.compute += interior_s.compute

    self.interior.update(interior_diff)

  def clear(self):
    self.interior.clear()
    self.frontier.clear()
    self.compute = 0

  def rebuild(self, tel : 'Telemetry'):
    assert self.base.material

    self.interior.clear()
    self.frontier.clear()
    self.compute = 0

    stack = [self.base]
    while stack:
      s = stack.pop()
      for t in s.tensors:
        for u in self.visitor(t):
          tel.summary['heuristic_access_count'] += 1
          us = u.storage
          us_id = us.root_id
          if us.material:
            if us_id != self.base.root_id:
              self.frontier.add(us_id)
            continue
          if us_id not in self.interior:
            self.compute += us.compute
            self.interior.add(us_id)
            stack.append(us)
