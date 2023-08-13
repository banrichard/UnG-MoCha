import torch
from typing import Union, Tuple, Callable

import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GATConv, GraphConv, SAGEConv, GINConv, GINEConv
from torch_geometric.nn.inits import reset, uniform, zeros


class NNGINConv(MessagePassing):
    """
    Add the node embedding with edge embedding
    """

    def __init__(self, edge_nn: Callable, node_nn: Callable,
                 eps: float = 0., train_eps: bool = False, aggr: str = 'add', **kwargs):
        super(NNGINConv, self).__init__(aggr=aggr, **kwargs)
        self.edge_nn = edge_nn
        self.node_nn = node_nn
        self.aggr = aggr
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def reset_parameters(self):
        reset(self.node_nn)
        reset(self.edge_nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        out = self.node_nn(out)
        # print(out.shape)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        edge_attr = self.edge_nn(edge_attr)
        return F.relu(x_j + edge_attr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.edge_nn,
                                   self.node_nn)


class NNGINConcatConv(MessagePassing):
    """
    Concatenate the node embedding with edge embedding
    no self loop
    """

    def __init__(self, edge_nn: Callable, node_nn: Callable,
                 eps: float = 0., train_eps: bool = False, aggr: str = 'add', **kwargs):
        super(NNGINConcatConv, self).__init__(aggr=aggr, **kwargs)
        self.edge_nn = edge_nn
        self.node_nn = node_nn
        self.aggr = aggr
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def reset_parameters(self):
        reset(self.node_nn)
        reset(self.edge_nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.node_nn(out)
        # print(out.shape)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        edge_attr = self.edge_nn(edge_attr)
        return torch.cat((x_j, edge_attr), dim=-1)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.edge_nn,
                                   self.node_nn)


class MotifGNN(nn.Module):
    def __init__(self, num_layers, num_g_hid, num_e_hid, out_g_ch, model_type, dropout, num_node_feat=1,
                 num_edge_feat=1, inter_hid=64):
        super(MotifGNN, self).__init__()
        self.num_node_feat = num_node_feat
        self.num_edge_feat = num_edge_feat
        self.num_layers = num_layers
        self.num_hid = num_g_hid
        self.num_e_hid = num_e_hid
        self.num_out = out_g_ch
        self.inter_hid = inter_hid
        self.model_type = model_type
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.agg = nn.Linear(self.num_out, self.inter_hid, bias=False)
        cov_layer = self.build_cov_layer(self.model_type)

        for l in range(self.num_layers):
            hidden_input_dim = self.num_node_feat if l == 0 else self.num_hid
            hidden_output_dim = self.num_out if l == self.num_layers - 1 else self.num_hid

            if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GAT" \
                    or self.model_type == "SAGE" or self.model_type == "GCN" \
                    or self.model_type == "DNA" or self.model_type == "Graph":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim))
            elif self.model_type == "FA":
                self.convs.append(cov_layer(hidden_input_dim, normalize=False))
            elif self.model_type == "NN" or self.model_type == "NNGIN" or self.model_type == "NNGINConcat":
                self.convs.append(cov_layer(hidden_input_dim, hidden_output_dim, self.num_e_hid))
            else:
                print("Unsupported model type!")
        self.reset_parameters()

    def build_cov_layer(self, model_type):
        if model_type == "GIN":
            return lambda in_ch, hid_ch: GINConv(nn=nn.Sequential(
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)))
        elif model_type == "GINE":
            return lambda in_ch, hid_ch: GINEConv(nn=nn.Sequential(
                nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)))
        elif model_type == "NN":
            return lambda in_ch, hid_ch, e_hid_ch: NNConv(in_ch, hid_ch,
                                                          nn=nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch),
                                                                           nn.ReLU(),
                                                                           nn.Linear(e_hid_ch, in_ch * hid_ch)))
        elif model_type == "NNGIN":
            return lambda in_ch, hid_ch, e_hid_ch: NNGINConv(
                edge_nn=nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch), nn.ReLU(), nn.Linear(e_hid_ch, in_ch)),
                node_nn=nn.Sequential(nn.Linear(in_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)))
        elif model_type == "NNGINConcat":
            return lambda in_ch, hid_ch, e_hid_ch: NNGINConcatConv(
                edge_nn=nn.Sequential(nn.Linear(self.num_edge_feat, e_hid_ch), nn.ReLU(),
                                      nn.Linear(e_hid_ch, e_hid_ch)),
                node_nn=nn.Sequential(nn.Linear(in_ch + e_hid_ch, hid_ch), nn.ReLU(), nn.Linear(hid_ch, hid_ch)))
        elif model_type == "GAT":
            return GATConv
        elif model_type == "SAGE":
            return SAGEConv
        elif model_type == "GCN":
            return GCNConv
        elif model_type == "Graph":
            return GraphConv
        else:
            print("Unsupported model type!")

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.agg.weight)

    def forward(self, x, edge_index, edge_attr=None,):
        x, edge_index, edge_attr = x.squeeze(0), edge_index.squeeze(0), edge_attr.squeeze(0)
        x = x.cuda(0)
        edge_index = edge_index.cuda(0)
        edge_attr = edge_attr.cuda(0)
        if self.model_type == 'FA' or self.model_type == 'GCN2':
            x_0 = x
        else:
            x_0 = None
        for i in range(self.num_layers):
            if self.model_type == "GIN" or self.model_type == "GINE" or self.model_type == "GAT" \
                    or self.model_type == "SAGE":
                x = self.convs[i](x, edge_index)  # for GIN and GINE
            elif self.model_type == "Graph" or self.model_type == "GCN":
                x = self.convs[i](x, edge_index, edge_weight=edge_attr)
            elif self.model_type == "FA" or self.model_type == "GCN2":
                x = self.convs[i](x, x_0, edge_index, edge_weight=edge_attr)
            elif self.model_type == "NN" or self.model_type == "NNGIN" or self.model_type == "NNGINConcat":
                x = self.convs[i](x=x, edge_index=edge_index, edge_attr=edge_attr)
            else:
                print("Unsupported model type!")

            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.unsqueeze(torch.sum(x, dim=0), dim=0)  # agg
        x = F.relu(self.agg(x))  # relu(Q*agg(x))
        return x
