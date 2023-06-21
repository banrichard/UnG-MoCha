import torch
import torch.nn as nn
from model.MLP import MLP
from model.Embedding import Embedding
from utils.batch import Batch
from model.GraphModel import GraphModel
from model.HUGNN import NestedGIN
from model.motifNet import MotifGNN


class EdgeMean(GraphModel):
    def __init__(self, config):
        super(EdgeMean, self).__init__(config)

        # self.ignore_norm = config["rgcn_ignore_norm"]
        self.predict_net_name = config['predict_net']
        # create networks
        # embed the node features and edge features
        p_emb_dim, g_emb_dim, p_e_emb_dim, g_e_emb_dim = self.get_emb_dim()
        # self.pre_g_enc = Embedding(config['init_g_dim'], g_emb_dim)
        self.g_net, g_dim = self.create_graph_net(
            hidden_dim=config["num_g_hid"],
            num_layers=config["graph_num_layers"], num_e_hid=128,
            dropout=self.dropout, model_type=config['graph_net'])

        self.p_net, p_dim = self.create_pattern_net(
            name="pattern", input_dim=p_emb_dim, hidden_dim=config["ppn_hidden_dim"],
            num_edge_feat=1,
            num_layers=config["ppn_pattern_num_layers"],
            dropout=self.dropout, model_type=config['motif_net'])
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
        self.config = config

    def create_graph_net(self, input_dim=128, **kwargs):
        num_layers = kwargs.get("num_layers", 3)
        hidden_dim = kwargs.get("num_g_hid", 128)
        e_hidden_dim = kwargs.get("num_e_hid", 128)
        dropout = kwargs.get("dropout", 0.2)
        model_type = kwargs.get("model_type", "GINE")
        out_dim = kwargs.get("out_dim", 64)
        net = NestedGIN(num_layers=num_layers, input_dim=input_dim, num_g_hid=hidden_dim, num_e_hid=e_hidden_dim,
                        model_type=model_type, out_dim=out_dim, dropout=dropout)
        return net, out_dim

    def create_pattern_net(self, input_dim, **kwargs):
        # num_layers, num_g_hid, num_e_hid, out_g_ch, model_type, dropout
        num_layers = kwargs.get("num_layers", 3)
        hidden_dim = kwargs.get("num_g_hid", 128)
        e_hidden_dim = kwargs.get("num_e_hid", 128)
        dropout = kwargs.get("dropout", 0.2)
        model_type = kwargs.get("model_type", "NNGINConcat")
        output_dim = kwargs.get("out_g_ch", 64)
        num_node_feat = kwargs.get("num_node_feat", 1)
        num_edge_feat = kwargs.get("num_edge_feat", 1)
        net = MotifGNN(num_layers=num_layers, num_g_hid=hidden_dim, num_e_hid=e_hidden_dim, dropout=dropout,
                       model_type=model_type, out_g_ch=output_dim, num_edge_feat=num_edge_feat,
                       num_node_feat=num_node_feat)
        return net, output_dim

    def forward(self, motif_x, motif_edge_index, motif_edge_attr, graph):
        pattern_emb = self.p_net(motif_x, motif_edge_index, motif_edge_attr)
        graph_output = self.g_net(graph)
        if self.predict_net_name.startswith("Film"):
            pred, var, filmreg = self.predict_net(pattern_emb, graph_output)
            # filmreg = (torch.sum(alpha ** 2)) ** 0.5 + (torch.sum(beta ** 2)) ** 0.5
            return pred, var, filmreg
        else:
            pred, var = self.predict_net(pattern_emb, graph_output)
            return pred, var
