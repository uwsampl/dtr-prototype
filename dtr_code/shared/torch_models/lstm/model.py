'''
This model is obtained from
https://github.com/jiangqy/LSTM-Classification-pytorch/blob/master/utils/LSTMClassifier.py
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# LSTMCell is implemented by ourselves. It is not related to other parts of the model in the original repository.
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

# This model is implemented by ourselves. It is not related to the model in the original repository.
# LSTM without embedding
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

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, use_gpu=True):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        embeds = torch.randint(self.vocab_size - 1, [sentence.size(0), self.batch_size, self.embedding_dim], dtype=torch.float).cuda()
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        return y

class GRUClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, dropout_embed=0.5, use_gpu=True):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.dropout = nn.Dropout(dropout_embed)

    def forward(self, sentence):
        hidden = Variable(torch.zeros(1, self.batch_size, self.hidden_dim)).cuda()
        x = self.word_embeddings(sentence)
        x = self.dropout(x)
        lstm_out, lstm_h = self.gru(x, hidden)
        x = F.tanh(torch.transpose(x, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        y  = self.hidden2label(x)
        return y
