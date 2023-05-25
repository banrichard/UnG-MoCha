import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool

from model.MLP import FC, MLP


class NestedGIN(torch.nn.Module):
    """
    Hierarchical GNN to embed the data graph
    """

    def __init__(self, num_layers, input_dim, num_g_hid, num_e_hid, model_type="GIN", dropout=0.2):
        super(NestedGIN, self).__init__()
        self.num_layers = num_layers
        self.num_hid = num_g_hid
        self.num_e_hid = num_e_hid
        self.model_type = model_type
        self.input_dim = input_dim
        self.dropout = dropout
        # self.mlp_in_ch = self.num_expert * self.out_g_ch if self.pool_type == "att" else self.out_g_ch
        self.convs = nn.ModuleList()
        cov_layer = self.build_cov_layer(self.model_type)

        for l in range(self.num_layers):
            hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
            hidden_output_dim = self.num_hid
            if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GAT":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))

        self.lin1 = nn.Linear(self.num_hid, self.num_hid)
        self.lin2 = nn.Linear(self.num_hid, self.num_hid)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform(self.lin1.weight)
        nn.init.xavier_uniform(self.lin2.weight)

    def build_conv_layers(self, model_type):
        if model_type == "GIN":
            return lambda in_ch, hid_ch: GINConv(nn=nn.Sequential(
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)), train_eps=True)
        elif model_type == "GINE":
            return lambda in_ch, hid_ch: GINEConv(nn=nn.Sequential(
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)), train_eps=True)

    def forward(self, data):
        edge_index, batch = data.edge_index, data.batch
        if 'x' in data:
            x = data.x
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)  # one-hot encoding
        xs = []
        for layer in range(len(self.convs)):
            x = self.convs[layer](x, edge_index)
            if layer == 0:
                xs = [x]
            else:
                xs += [x]
            if layer < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(torch.cat(xs, dim=1), batch)
        # final graph representation
        x = global_add_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
