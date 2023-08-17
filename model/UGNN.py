import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
import math

from model.MLP import MLP


class UGNN(MessagePassing):
    def __init__(self, n_in_dim=128, e_in_dim=2, hid_dim=128, out_dim=128):
        super().__init__()
        self.n_in_dim = n_in_dim
        self.e_in_dim = e_in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.mlp = MLP(self.e_in_dim, self.hid_dim, self.n_in_dim)

        self.mlp.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # edge_attr is the edge score \in [edge_prob,edge_freq]
        edge_score = self.mlp(edge_attr)
        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_score)

    def message(self, x_j: Tensor, edge_attr) -> Tensor:
        num_neighbors = x_j.size(0)
        num_sample = math.ceil(num_neighbors * random.random())

        # how to randomly sample within the sample range?
        neighbor_idx = torch.randperm(num_neighbors)[:num_sample]
        x_j = x_j[neighbor_idx]
        return x_j


if __name__ == "__main__":
    in_channels = 64
    out_channels = 128
    num_nodes = 10
    num_edges = 20
    x = torch.rand(num_nodes, in_channels)
    edge_index = torch.randint(num_nodes, (2, num_edges)).to(torch.long)
    edge_attr = torch.randn(num_edges, 2)
    model = UGNN(n_in_dim=in_channels,e_in_dim=2,hid_dim=128, out_dim=out_channels)
    output = model(x, edge_index=edge_index, edge_attr=edge_attr)
    print(output)
