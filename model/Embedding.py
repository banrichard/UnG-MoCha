import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.emb_layer = nn.Linear(input_dim, emb_dim, bias=True)

    def forward(self, x):
        out = self.emb_layer(x)
        return out


