import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable



class EmbeddingLayer(nn.Module):

    class EmbeddingLayer(nn.Module):
        def __init__(self, vocab_size, embed_dim, drop_embed=0.25):
            super(EmbeddingLayer, self).__init__()
            self.drop = nn.Dropout(drop_embed)
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            # self.init_embedding_weights(vocab_size, embed_dim)

        def forward(self, input_variable):
            pass #todo



class Encoder(nn.Module):
    def __init__(self, bsize, embed_dim, encod_dim, model, n_layers, drop_encoder=0):
        super(Encoder, self).__init__()
        self.bsize = bsize
        self.word_emb_dim = embed_dim
        self.enc_lstm_dim = encod_dim
        self.dpout_model = drop_encoder
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")