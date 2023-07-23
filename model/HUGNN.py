import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool
from model.topK import TopKEdgePooling


class NestedGIN(torch.nn.Module):
    """
    Hierarchical GNN to embed the data graph
    """

    def __init__(self, num_layers, input_dim=128, num_g_hid=128, num_e_hid=128, out_dim=64, model_type="GIN",
                 dropout=0.2):
        super(NestedGIN, self).__init__()
        self.num_layers = num_layers
        self.num_hid = num_g_hid
        self.num_e_hid = num_e_hid
        self.model_type = model_type
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dropout = dropout
        # self.mlp_in_ch = self.num_expert * self.out_g_ch if self.pool_type == "att" else self.out_g_ch
        self.convs = nn.ModuleList()
        cov_layer = self.build_conv_layers(model_type)
        self.pooling = TopKEdgePooling(in_channels=self.input_dim, ratio=None, min_score=0.3)
        for l in range(self.num_layers):
            hidden_input_dim = self.input_dim if l == 0 else self.num_hid
            hidden_output_dim = self.num_hid
            if self.model_type == "GIN" or self.model_type == "GINE":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
        self.lin1 = nn.Linear(self.num_hid * self.num_layers, self.num_hid)
        self.lin2 = nn.Linear(self.num_hid, self.out_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for layer in [self.lin1, self.lin2]:
            nn.init.xavier_uniform(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_uniform_(self.pooling.weight)

    def build_conv_layers(self, model_type):
        if model_type == "GIN":
            return lambda in_ch, hid_ch: GINConv(nn=nn.Sequential(
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)), train_eps=True)
        elif model_type == "GINE":
            return lambda in_ch, hid_ch: GINEConv(nn=nn.Sequential(
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)), edge_dim=self.num_e_hid,
                train_eps=True)

    def forward(self, data):
        data = data.cuda()
        edge_index, edge_attr, batch, edge_batch = data.edge_index, data.edge_attr, data.node_to_subgraph, data.edge_to_subgraph
        edge_attr = edge_attr.view(-1, 1).expand(-1, self.num_e_hid)
        if 'x' in data:
            x = data.x.cuda()
        else:
            x = torch.zeros([edge_index.max() + 1, 1])
            x = x.cuda()
        x, edge_index, edge_attr, batch = self.pooling(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch,
                                                       edge_batch=edge_batch)
        xs = []
        for layer in range(len(self.convs)):
            if self.model_type == "GIN":
                x = self.convs[layer](x=x, edge_index=edge_index)
            elif self.model_type == "GINE":
                x = self.convs[layer](x=x, edge_index=edge_index, edge_attr=edge_attr)
            if layer < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if layer == 0:
                xs = [x]
            else:
                xs += [x]
        x = torch.cat(xs, dim=1)
        num_nodes = x.size(0)
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[edge_index] = 1
        # Apply mask
        x = x[mask]
        batch = batch[mask]
        x = global_mean_pool(x, batch)
        # final graph representation
        x = global_add_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
