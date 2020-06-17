'''
Adapted from https://github.com/mk-minchul/unroll_gan
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        # self.activation_fn = torch.tanh
        self.activation_fn = F.relu

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return self.map3(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.activation_fn = F.relu

    def forward(self, x):
        x = self.activation_fn(self.map1(x))
        x = self.activation_fn(self.map2(x))
        return torch.sigmoid(self.map3(x))

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()