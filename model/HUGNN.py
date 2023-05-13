import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv,global_mean_pool, global_add_pool

from model.MLP import FC


class NestedGIN(torch.nn.Module):
    def __init__(self, args, num_node_feat, num_edge_feat):
        super(NestedGIN, self).__init__()
        self.num_layers = args.num_layers
        self.num_hid = args.num_g_hid
        self.num_e_hid = args.num_e_hid
        self.num_out = args.out_g_ch
        self.model_type = args.model_type
        self.dropout = args.dropout
        self.num_expert = args.num_expert
        self.out_g_ch = args.out_g_ch
        self.mlp_in_ch = self.num_expert * self.out_g_ch if self.pool_type == "att" else self.out_g_ch
        self.num_mlp_hid = args.num_mlp_hid
        self.convs = nn.ModuleList()
        cov_layer = self.build_cov_layer(self.model_type)

        for l in range(self.num_layers):
            hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
            hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid

            if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GAT":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
        self.lin1 = nn.Linear(self.num_hid, self.num_hid)
        self.lin2 = nn.Linear(self.num_hid, self.num_hid)
        self.fc_hid = FC(in_ch=self.mlp_in_ch, out_ch=self.num_mlp_hid)
        self.fc_reg = FC(in_ch=self.num_mlp_hid, out_ch=1)
        # add a fc layer to calculate the variance
        self.fc_var = FC(in_ch=self.num_mlp_hid, out_ch=1)
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.fc_hid.reset_parameters()
        self.fc_reg.reset_parameters()
        self.fc_var = FC(in_ch=self.num_mlp_hid, out_ch=1)
    def build_conv_layers(self,model_type):
        if model_type == "GIN":
            return lambda in_ch, hid_ch: GINConv(nn=nn.Sequential(
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)))
        elif model_type == "GINE":
            return lambda in_ch, hid_ch: GINEConv(nn=nn.Sequential(
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)))
    def forward(self, data):
        edge_index, batch = data.edge_index, data.batch
        if 'x' in data:
            x = data.x
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device) # one-hot encoding
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_add_pool(x, data.node_to_subgraph)
        x = global_add_pool(x, data.subgraph_to_graph)
        hid_g = F.relu(self.fc_hid(x))
        output = self.fc_reg(hid_g)
        output_var = self.fc_var(hid_g)
        return output,output_var
