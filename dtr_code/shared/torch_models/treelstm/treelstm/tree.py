# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self, x=None):
        self.x = x
        self.num_children = 0
        self.children = list()
    
    def __init__(self, x, children):
        self.x = x
        self.children = children
        self.num_children = len(children)
        self.checkpointed = False
    
    def map_(self, f):
        self.checkpointed = True
        self.x = f(self.x)
        list(map(lambda ch: ch.map_(f), self.children))

    def add_child(self, child):
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if hasattr(self, '_size') and getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth
