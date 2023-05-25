import torch
import torch.nn as nn

from model.GraphModel import GraphModel
from model.HUGNN import NestedGIN
from model.motifNet import MotifGNN
from PredictNet import PatternReadout, GraphReadout


def split_batch(index, input, length, max):
    device = input.device
    input = torch.cat([torch.zeros([1, input.size(1)], device=device), input], dim=0)
    return input[index]


class EdgeMean(GraphModel):
    def __init__(self, config):
        super(EdgeMean, self).__init__(config)

        # self.ignore_norm = config["rgcn_ignore_norm"]

        # create networks
        # embed the node features and edge features
        p_emb_dim, g_emb_dim, p_e_emb_dim, g_e_emb_dim = self.get_emb_dim()
        # TODO: modify create_net to adapt the NNGIN and NestedGIN
        self.g_net, g_dim = self.create_graph_net(
            input_dim=g_emb_dim * 3, hidden_dim=config["num_g_hid"],
            num_layers=config["graph_num_layers"], num_e_hid=128,
            dropout=self.dropout, model_type="GIN", bsz=config["batch_size"])

        self.p_net, p_dim = self.create_pattern_net(
            name="pattern", input_dim=p_emb_dim * 3, hidden_dim=config["ppn_hidden_dim"],
            num_layers=config["ppn_pattern_num_layers"], act_func=self.act_func,
            dropout=self.dropout, bsz=config["batch_size"])
        # create predict layers

        if self.add_enc:
            p_enc_dim, g_enc_dim, p_e_enc_dim, g_e_enc_dim = self.get_enc_dim()
            p_dim += p_enc_dim * 2 + p_e_enc_dim
            g_dim += g_enc_dim * 2 + g_e_enc_dim
        if self.add_degree:
            p_dim += 2
            g_dim += 2
        self.predict_net = self.create_predict_net(config["predict_net"],
                                                   pattern_dim=p_dim, graph_dim=g_dim,
                                                   hidden_dim=config["predict_net_hidden_dim"])

        self.g_linear = torch.nn.Linear(g_emb_dim * 3, config["ppn_hidden_dim"])
        self.p_linear = torch.nn.Linear(p_emb_dim * 3, config["ppn_hidden_dim"])
        self.config = config

    def create_graph_net(self, input_dim, **kwargs):
        num_layers = kwargs.get("num_layers", 1)
        hidden_dim = kwargs.get("num_g_hid", 128)
        e_hidden_dim = kwargs.get("num_e_hid", 128)
        dropout = kwargs.get("dropout", 0.2)
        model_type = kwargs.get("model_type", "GIN")
        net = NestedGIN(num_layers=num_layers, input_dim=input_dim, num_g_hid=hidden_dim, num_e_hid=e_hidden_dim,
                        model_type=model_type,
                        dropout=dropout)
        return net, hidden_dim

    def create_pattern_net(self, input_dim, **kwargs):
        # num_layers, num_g_hid, num_e_hid, out_g_ch, model_type, dropout
        num_layers = kwargs.get("num_layers", 1)
        hidden_dim = kwargs.get("num_g_hid", 128)
        e_hidden_dim = kwargs.get("num_e_hid", 128)
        dropout = kwargs.get("dropout", 0.2)
        model_type = kwargs.get("model_type", "NNGINConcat")
        output_dim = kwargs.get("out_g_ch", 64)
        num_edge_feat = kwargs.get("num_edge_feat", 1)
        net = MotifGNN(num_layers=num_layers, num_g_hid=hidden_dim, num_e_hid=e_hidden_dim, dropout=dropout,
                       model_type=model_type, out_g_ch=output_dim, num_edge_feat=num_edge_feat, num_node_feat=input_dim)
        return net, hidden_dim

    def GraphEmbedding(self, g_vl_emb, g_el_emb, adj):
        u = g_vl_emb[adj[0]]
        v = g_vl_emb[adj[1]]
        result = torch.cat([u, g_el_emb, v], dim=1)  # x_u || e_<u,v> || x_v
        return result

    def PredictEnc(self, edge_enc, pattern_enc, adj):
        u = pattern_enc[adj[0]]
        v = pattern_enc[adj[1]]
        result = torch.cat([u, edge_enc, v], dim=1)
        return result

    def CatIndeg(self, indeg, adj):
        u = indeg[adj[0]]
        v = indeg[adj[1]]
        result = torch.cat([u, v], dim=1)
        return result

    def forward(self, motif_x, motif_edge_index, motif_edge_attr, graph):
        zero_mask = None

        p_emb, g_vl_emb, p_el_emb, g_el_emb = self.get_emb(motif_edge_attr, graph)

        # graph_output.masked_fill_(zero_output_mask, 0.0)
        # graph_output = graph_output.resize(graph_output.size(0) * graph_output.size(1), graph_output.size(2))
        pattern_emb = self.p_net(motif_x, motif_edge_index, motif_edge_attr)
        graph_output = self.g_net(graph)
        pred, alpha, beta = self.predict_net(pattern_emb, graph_output)
        filmreg = (torch.sum(alpha ** 2)) ** 0.5 + (torch.sum(beta ** 2)) ** 0.5
        return pred, filmreg
