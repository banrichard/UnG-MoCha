import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):

    def __init__(self, input_dim, emb_dim):
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.emb_layer = nn.Linear(input_dim, emb_dim, bias=False)
        nn.init.normal_(self.emb_layer.weight, 0.0, 1.0)

    def forward(self, x):
        out = self.emb_layer(x)
        return out
