'''
LSTM RNN we wrote ourselves
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class LSTMCell(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super().__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim, bias=False)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
    def forward(self, input, children):
        """input is a single input, while
        children is a list of tuple of memory."""
        iou = self.ioux(input)
        if len(children) > 0:
            s = children[0][0]
            for i in range(1, len(children)):
                s = s + children[i][0]
            iou = iou + self.iouh(s)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)
        c = i * u
        fx = self.fx(input)
        for _, child in children:
            c += F.sigmoid(fx * self.fh(child))
        h = o * F.tanh(c)
        return c, h

class LSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, use_dtr):
        super().__init__()
        self.cell = LSTMCell(in_dim, mem_dim)
        self.use_dtr = use_dtr

    def forward(self, inputs):
        output = []
        c = torch.zeros([1, self.cell.mem_dim]).cuda()
        h = torch.zeros([1, self.cell.mem_dim]).cuda()
        if self.use_dtr:
            c = c.detach().checkpoint()
            h = h.detach().checkpoint()
        for i in inputs:
            c, h = self.cell(i, [(c, h)])
            output.append(c)
        return output
