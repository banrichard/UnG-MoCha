import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, GINEConv,global_mean_pool, global_add_pool

class HUGNN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super().__init__()
        self.dataset = dataset
        self.num_layers = num_layers
        self.hidden = hidden
        input_dim = dataset.num_features