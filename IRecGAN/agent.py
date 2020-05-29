import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from main.layers import EmbeddingLayer, Encoder

class Agent(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, numlabel, n_layers, recommendation=100, feature_vec=None, init=False,
                 model='LSTM'):
        super(Agent, self).__init__()
        # classifier
        self.batch_size = bsize
        self.nonlinear_fc = False
        self.n_classes = numlabel
        self.enc_lstm_dim = encod_dim
        self.encoder_type = 'Encoder'
        self.model = model
        self.gamma = 0.9
        self.n_layers = n_layers
        self.recommendation = recommendation #Only top 10 items are selected
        self.embedding=EmbeddingLayer(numlabel, embed_dim)

        self.encoder = eval(self.encoder_type)(self.batch_size, embed_dim, self.enc_lstm_dim, self.model, self.n_layers)
        self.enc2out = nn.Linear(self.enc_lstm_dim, self.n_classes)
        if init:
            self.init_params()

    def copy_weight(self, feature_vector):
        self.embedding.init_embedding_weights(feature_vector)

    def init_params(self):
        for param in self.parameters():
            init.normal_(param, 0, 1)

    def forward(self, sequence, evaluate = True):
        seq_em = sequence
        seq_len = len(sequence)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder((seq_em, seq_len))
        else:
            enc_out, h = self.encoder((seq_em, seq_len))
        output = self.enc2out(enc_out[:, -1, :]) #batch*hidden
        output = F.softmax(output, dim = 1)
        # indices is with size of batch_size * self.recommendation
        if evaluate:
            _, indices = torch.topk(output, self.recommendation, dim = 1, sorted = True)
        else:
            indices = torch.multinormial(output, self.recommendation)
        if self.model == 'LSTM':
            return output, indices, (h, c)
        else:
            return output, indices, h


    def step(self, click, hidden, evaluate = True):
        seq_em = self.embedding(click)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder((seq_em, hidden))
        else:
            enc_out, h = self.encoder((seq_em, hidden))

        output = self.enc2out(enc_out[:, -1, :]) #batch*hidden
        output = F.softmax(output, dim = 1)
        if not evaluate:
            indices = torch.multinormial(output, self.recommendation)
        else:
            _, indices = torch.topk(output, self.recommendation, dim = 1, sorted = True)
        #select from only top k
        if self.model == 'LSTM':
            return output, indices, (h, c)
        else:
            return output, indices, h











