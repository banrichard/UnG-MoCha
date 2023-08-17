import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GINEConv, global_mean_pool, global_add_pool
from model.topK import TopKEdgePooling


class NestedGNN(torch.nn.Module):
    """
    Hierarchical GNN to embed the data graph
    """

    def __init__(self, num_layers, input_dim=128, num_g_hid=128, num_e_hid=128, out_dim=64, model_type="GIN",
                 dropout=0.2, gsl=True):
        super(NestedGNN, self).__init__()
        self.num_layers = num_layers
        self.num_hid = num_g_hid
        self.num_e_hid = num_e_hid
        self.model_type = model_type
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.gsl = gsl
        # self.mlp_in_ch = self.num_expert * self.out_g_ch if self.pool_type == "att" else self.out_g_ch
        self.convs = nn.ModuleList()
        cov_layer = self.build_conv_layers(model_type)
        self.pooling = TopKEdgePooling(ratio=None, min_score=0.3)
        for l in range(self.num_layers):
            hidden_input_dim = self.input_dim if l == 0 else self.num_hid
            hidden_output_dim = self.num_hid
            if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GCN" or self.model_type == "GAT" or self.model_type == "GraphSage":
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
        elif model_type == "GCN":
            return GCNConv
        elif model_type == "GAT":
            return GATConv
        elif model_type == "GraphSage":
            return SAGEConv

    def forward(self, data):
        edge_index, edge_attr, batch, edge_batch = data.edge_index.cuda(), data.edge_attr.cuda(), data.batch.cuda(), data.edge_batch.cuda()
        # edge_attr = edge_attr.view(-1, 1).expand(-1, self.num_e_hid)
        if 'x' in data:
            x = data.x.cuda()
        else:
            x = torch.zeros([edge_index.max() + 1, 1]).cuda()
        if self.gsl:
            x, edge_index, edge_attr, batch = self.pooling(data)
        edge_attr = edge_attr[:, 0].view(-1, 1).expand(-1, self.num_e_hid)
        xs = []
        for layer in range(len(self.convs)):
            if self.model_type == "GIN" or self.model_type == "GCN" or self.model_type == "GraphSage":
                x = self.convs[layer](x=x, edge_index=edge_index)
            elif self.model_type == "GINE" or self.model_type == "GAT":
                x = self.convs[layer](x=x, edge_index=edge_index, edge_attr=edge_attr)
            if layer < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if layer == 0:
                xs = [x]
            else:
                xs += [x]
        x = torch.cat(xs, dim=1)
        x = global_mean_pool(x, batch)
        # final graph representation
        x = global_add_pool(x, torch.zeros(batch.max() + 1).to(torch.long).to(x.device))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        return x
