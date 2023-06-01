import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool

from model.MLP import FC, MLP


class NestedGIN(torch.nn.Module):
    """
    Hierarchical GNN to embed the data graph
    """

    def __init__(self, num_layers, input_dim, num_g_hid, num_e_hid, out_dim=64, model_type="GINE", dropout=0.2):
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

        for l in range(self.num_layers):
            hidden_input_dim = self.input_dim if l == 0 else self.num_hid
            hidden_output_dim = self.num_hid
            if self.model_type == "GIN" or self.model_type == "GINE":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))

        self.lin1 = nn.Linear(self.num_hid, self.num_hid)
        self.lin2 = nn.Linear(self.num_hid, self.out_dim)

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
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)), edge_dim=hid_ch, train_eps=True)

    def forward(self, data):
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.node_to_subgraph
        if 'x' in data:
            x = data.x
        else:
            x = torch.ones([edge_index.max() + 1, 1])
        xs = []
        for layer in range(len(self.convs)):
            x = self.convs[layer](x=x, edge_index=edge_index)
            if layer == 0:
                xs = [x]
            else:
                xs += [x]
            if layer < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        # final graph representation
        x = global_add_pool(x, data.subgraph_to_graph)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class NestedGIN_eff(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, out_dim, graph_pred=True,
                 dropout=0.2, multi_layer=False, edge_nest=True):
        super(NestedGIN_eff, self).__init__()
        self.graph_pred = graph_pred  # delete the final graph-level pooling
        self.dropout = dropout  # dropout 0.1 for multilayer, 0.2 for no multi
        self.multi_layer = multi_layer  # to use multi layer supervision or not
        self.edge_nest = edge_nest  # denote whether using the edge-level nested information
        z_in = 1800  # if self.use_rd else 1700
        self.out_dim = out_dim
        emb_dim = hidden
        self.z_initial = torch.nn.Embedding(z_in, emb_dim)
        self.z_embedding = nn.Sequential(nn.Dropout(dropout),
                                         nn.BatchNorm1d(emb_dim),
                                         nn.ReLU(),
                                         nn.Linear(emb_dim, emb_dim),
                                         nn.Dropout(dropout),
                                         nn.BatchNorm1d(emb_dim),
                                         nn.ReLU()
                                         )
        input_dim = 1  # 1800#dataset.num_features
        # if self.use_z or self.use_rd:
        #    input_dim += 8
        self.x_embedding = nn.Sequential(nn.Linear(input_dim, hidden),
                                         nn.Dropout(dropout),
                                         nn.BatchNorm1d(hidden),
                                         nn.ReLU(),
                                         nn.Linear(hidden, hidden),
                                         nn.Dropout(dropout),
                                         nn.BatchNorm1d(hidden),
                                         nn.ReLU()
                                         )

        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            ),
            train_eps=True,
            edge_dim=hidden)

        # self.conv1 = GATConv(input_dim, hidden, edge_dim = hidden, add_self_loops = False)

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GINEConv(
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                ),
                train_eps=True,
                edge_dim=hidden))

            # self.convs.append(GATConv(hidden, hidden, edge_dim = hidden, add_self_loops = False))

        self.lin1 = nn.Linear(num_layers * hidden + hidden, hidden)
        # self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin1 = torch.nn.Linear(hidden, hidden)
        self.bn_lin1 = nn.BatchNorm1d(hidden, eps=1e-5, momentum=0.1)
        self.lin2 = nn.Linear(hidden, self.out_dim)

        # self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for layer in self.z_embedding.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.bn_lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        data.to(self.lin1.weight.device)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # edge_pos[:, :200] = 0
        # edge_pos[:, 200:500] = 0
        # edge_pos[:, -1300:] = 0

        if hasattr(data, 'edge_pos'):
            # original, slow version
            edge_pos = data.edge_pos.float()
            z_emb = torch.mm(edge_pos, self.z_initial.weight)
        else:
            # new, fast version

            # for ablation study
            # mask_index = (data.pos_index >= 500)
            # mask_index = torch.logical_and((data.pos_index >= 200), (data.pos_index < 500))
            # mask_index = (data.pos_index < 500)
            # z_emb = global_add_pool(torch.mul(self.z_initial.weight[data.pos_index[~mask_index]], data.pos_enc[~mask_index].view(-1, 1)), data.pos_batch[~mask_index])

            z_emb = global_add_pool(torch.mul(self.z_initial.weight[data.pos_index], data.pos_enc.view(-1, 1)),
                                    data.pos_batch)
        z_emb = self.z_embedding(z_emb)

        # z_emb = self.z_embedding(edge_pos)

        if self.use_id is None:
            x = self.conv1(x, edge_index, z_emb)

        # xs = [x]
        xs = [self.x_embedding(data.x), x]
        for conv in self.convs:
            x = conv(x, edge_index, z_emb)
            if self.edge_nest:
                x = conv(x, edge_index, data.node_id) + conv(x, edge_index, data.node_id + 1)
            else:
                x = conv(x, edge_index, data.node_id)
            xs += [x]

        if self.graph_pred:
            # x = global_add_pool(x, data.batch)
            x = global_mean_pool(torch.cat(xs, dim=1), batch)
        else:
            x = torch.cat(xs, dim=1)
            # x = x
        x = self.lin1(x)
        if x.size()[0] > 1:
            x = self.bn_lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        # # if not self.use_cycle:
        # #     return F.log_softmax(x, dim=-1)
        # else:
        return x  # , []
