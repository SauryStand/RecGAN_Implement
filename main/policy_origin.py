import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init    
from nn_layer_simu import EmbeddingLayer, Encoder
    
class Policy(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, numlabel, recom_number=50, feature_vec=None, init=False, model='LSTM'):
        super(Policy, self).__init__()
        # classifier
        self.batch_size = bsize
        self.nonlinear_fc = False
        self.n_classes = numlabel
        self.enc_lstm_dim = encod_dim
        self.encoder_type = 'Encoder'
        self.model = model
        self.recom = recom_number
        self.embedding=EmbeddingLayer(numlabel, embed_dim)
        self.encoder = eval(self.encoder_type)(self.batch_size, embed_dim, self.enc_lstm_dim, self.model)
        self.enc2out = nn.Linear(self.enc_lstm_dim, self.n_classes)

    def init_params(self):
        for param in self.parameters():
            init.normal_(param, 0, 1)
    def forward(self, seq, evaluate=False):
        # seq : (seq, seq_len)
        seq_em, seq_len= seq
        seq_em = self.embedding(seq_em)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder((seq_em, seq_len))
        else:
            enc_out, h = self.encoder((seq_em, seq_len))
        output = F.softmax(self.enc2out(h.squeeze(0)), dim=1)
         
        if evaluate:
            _, indices = torch.topk(output, self.recom, dim = 1)
        else:
            indices = torch.multinomial(output, self.recom)

        if self.model == 'LSTM':
            return output, indices, (h, c)
        else:
            return output, indices, h
        
    def step(self, click, hidden, evaluate=False):
        seq_em = self.embedding(click)
        if self.model == 'LSTM':
            enc_out, (h, c) = self.encoder.step_cell(seq_em, hidden)
        else:
            enc_out, h = self.encoder.step_cell(seq_em, hidden)
        output = F.softmax(self.enc2out(h.squeeze(0)), dim=1)
         
        if evaluate:
            _, indices = torch.topk(output, self.recom, dim = 1)
        else:
            indices = torch.multinomial(output, self.recom)
        if self.model == 'LSTM':
            return output, indices, (h, c)
        else:
            return output, indices, h








