import numpy as np
import torch
import torch.nn as nn

from model.UGNN import Embedding
from model.MLP import MLP
from model.PredictNet import FilmSumPredictNet, DIAMNet, MeanAttnPredictNet, MeanPredictNet, CCANet
from utils.utils import get_enc_len, int2onehot


class GraphModel(nn.Module):
    def __init__(self, config):
        # motif_x, motif_edge_index, motif_edge_attr, graph
        super(GraphModel, self).__init__()
        self.act_func = config["activation_function"]
        self.init_emb = config["init_emb"]
        self.share_emb = config["share_emb"]
        self.share_arch = config["share_arch"]
        self.base = config["base"]
        self.max_ngv = config["max_ngv"]
        self.max_nge = config["max_nge"]
        self.max_ngel = config["max_ngel"]
        self.max_npv = config["max_npv"]
        self.max_npe = config["max_npe"]
        self.max_npel = config["max_npel"]

        self.emb_dim = config["emb_dim"]
        self.dropout = config["dropout"]
        self.dropatt = config["dropatt"]
        self.add_enc = config["predict_net_add_enc"]
        self.add_degree = config["predict_net_add_degree"]

        # create encoding layer
        # self.g_v_enc, self.g_e_enc, self.g_el_enc = \
        #     [self.create_enc(max_n, self.base) for max_n in [self.max_ngv, self.max_nge, self.max_ngel]]
        # self.p_v_enc, self.p_e_enc, self.p_el_enc = [self.create_enc(max_n, self.base) for max_n in
        #                                              [self.max_npv, self.max_npe,
        #                                               self.max_npel]]
        #
        # # create embedding layers
        # self.g_v_emb = self.create_emb(self.g_v_enc.embedding_dim, self.emb_dim, init_emb=self.init_emb)
        # self.g_el_emb = self.create_emb(self.g_el_enc.embedding_dim, self.emb_dim, init_emb=self.init_emb)
        #
        # self.p_el_emb = self.create_emb(self.p_el_enc.embedding_dim, self.emb_dim, init_emb=self.init_emb)

        # create networks
        # create predict layers

    def create_enc(self, max_n, base):
        enc_len = get_enc_len(max_n - 1, base)
        enc_dim = enc_len * base
        enc = nn.Embedding(max_n, enc_dim)
        enc.weight.data.copy_(torch.from_numpy(int2onehot(np.arange(0, max_n), enc_len, base)))
        enc.weight.requires_grad = False
        return enc

    def create_emb(self, input_dim, emb_dim, init_emb=True):
        if not init_emb:
            emb = None
        else:
            emb = Embedding(input_dim, emb_dim)
        return emb

    def get_enc_dim(self):
        g_dim = self.base * get_enc_len(self.max_ngv - 1, self.base)
        g_e_dim = self.base * (
                get_enc_len(self.max_nge - 1, self.base) + get_enc_len(self.max_ngel - 1, self.base))
        p_dim = self.base * get_enc_len(self.max_npv - 1, self.base)
        p_e_dim = self.base * (
                get_enc_len(self.max_npe - 1, self.base) + get_enc_len(self.max_npel - 1, self.base))
        return p_dim, g_dim, p_e_dim, g_e_dim

    def get_enc(self, pattern, graph):

        pattern_v = self.p_v_enc(pattern.x)
        pattern_el = self.p_el_enc(pattern.edge_attr)
        graph_v = self.g_v_enc(graph.x)
        graph_el = self.g_el_enc(graph.edge_attr)

        p_enc = pattern_v
        p_e_enc = pattern_el
        g_enc = graph_v
        g_e_enc = graph_el
        return p_enc, g_enc, p_e_enc, g_e_enc

    def get_emb(self, pattern, graph):
        # bsz = pattern_len.size(0)

        # p_v_enc,p_vl_enc通过create_enc调用nn.Embedding()网络,将词转化为词向量
        # 但其实pattern_v,graph_v并没有用到
        # pattern_v = self.p_v_enc(pattern.x)
        pattern_e = self.p_e_enc(pattern.edge_attr)
        # graph_v = self.g_v_enc(graph.x)
        graph_el = self.g_el_enc(graph.edge_attr)

        if self.init_emb == "None":
            # g_emb = graph_v
            g_e_emb = graph_el

        else:
            # p_vl_emb将词向量通过create_emb做了一个线性映射，有三种不同的方法，区别在于线性映射的参数初始化方式不同
            p_e_emb = self.p_el_emb(pattern_e)
            g_e_emb = self.g_el_emb(graph_el)
        return p_e_emb, g_e_emb

    def get_emb_dim(self):
        if self.init_emb == "None":
            return self.get_enc_dim()
        else:
            return self.emb_dim, self.emb_dim, self.emb_dim, self.emb_dim

    def increase_input_size(self, config):
        super(GraphModel, self).increase_input_size(config)

        # create encoding layers
        new_g_v_enc, new_g_vl_enc = \
            [self.create_enc(max_n, self.base) for max_n in [config["max_ngv"], config["max_ngvl"]]]
        if self.share_emb:
            new_p_v_enc, new_p_vl_enc = \
                new_g_v_enc, new_g_vl_enc
        else:
            new_p_v_enc, new_p_vl_enc = \
                [self.create_enc(max_n, self.base) for max_n in [config["max_npv"], config["max_npvl"]]]
        del self.g_v_enc, self.g_vl_enc
        del self.p_v_enc, self.p_vl_enc
        self.g_v_enc, self.g_vl_enc = new_g_v_enc, new_g_vl_enc
        self.p_v_enc, self.p_vl_enc = new_p_v_enc, new_p_vl_enc

        # increase embedding layers
        self.g_vl_emb.increase_input_size(self.g_vl_enc.embedding_dim)
        if not self.share_emb:
            self.p_vl_emb.increase_input_size(self.p_vl_enc.embedding_dim)

        # increase networks

        # increase predict network

        # set new parameters
        # npv:pattern vertex
        # npvl:pattern vertex label
        # npe:pattern edge
        # npel:pattern edge label
        self.max_npv = config["max_npv"]
        self.max_npvl = config["max_npvl"]
        self.max_npe = config["max_npe"]
        self.max_npel = config["max_npel"]
        self.max_ngv = config["max_ngv"]
        self.max_ngvl = config["max_ngvl"]
        self.max_nge = config["max_nge"]
        self.max_ngel = config["max_ngel"]

    def create_predict_net(self, predict_type, pattern_dim, graph_dim, **kw):
        if predict_type == "None":
            predict_net = None
        elif predict_type == "MLP":
            hidden_dim = kw.get("hidden_dim", 64)
            predict_net = MLP(pattern_dim + graph_dim, hidden_dim, 1)
        elif predict_type == "MeanPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            predict_net = MeanPredictNet(pattern_dim, graph_dim, hidden_dim,
                                         dropout=self.dropout)
        elif predict_type == "SumPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            predict_net = SumPredictNet(pattern_dim, graph_dim, hidden_dim,
                                        act_func=self.act_func, dropout=self.dropout)
        # elif predict_type == "FilmSumPredictNet":
        #     hidden_dim = kw.get("hidden_dim", 64)
        #     predict_net = FilmSumPredictNet(pattern_dim, graph_dim, hidden_dim,
        #                                     act_func=self.act_func, dropout=self.dropout)
        elif predict_type == "FilmSumPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            predict_net = FilmSumPredictNet(pattern_dim, graph_dim, hidden_dim,
                                            dropout=self.dropout)
        elif predict_type == "MaxPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            predict_net = MaxPredictNet(pattern_dim, graph_dim, hidden_dim,
                                        act_func=self.act_func, dropout=self.dropout)
        elif predict_type == "MeanAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 4)
            predict_net = MeanAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                                             num_heads=num_heads, recurrent_steps=recurrent_steps,
                                             dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "SumAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            predict_net = SumAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                                            act_func=self.act_func,
                                            num_heads=num_heads, recurrent_steps=recurrent_steps,
                                            dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "MaxAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            predict_net = MaxAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                                            act_func=self.act_func,
                                            num_heads=num_heads, recurrent_steps=recurrent_steps,
                                            dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "MeanMemAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            mem_len = kw.get("mem_len", 4)
            predict_net = MeanMemAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                                                act_func=self.act_func,
                                                num_heads=num_heads, recurrent_steps=recurrent_steps,
                                                mem_len=mem_len,
                                                dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "SumMemAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            mem_len = kw.get("mem_len", 4)
            predict_net = SumMemAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                                               act_func=self.act_func,
                                               num_heads=num_heads, recurrent_steps=recurrent_steps,
                                               mem_len=mem_len,
                                               dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "MaxMemAttnPredictNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 1)
            mem_len = kw.get("mem_len", 4)
            predict_net = MaxMemAttnPredictNet(pattern_dim, graph_dim, hidden_dim,
                                               act_func=self.act_func,
                                               num_heads=num_heads, recurrent_steps=recurrent_steps,
                                               mem_len=mem_len,
                                               dropout=self.dropout, dropatt=self.dropatt)
        elif predict_type == "DIAMNet":
            hidden_dim = kw.get("hidden_dim", 64)
            recurrent_steps = kw.get("recurrent_steps", 1)
            num_heads = kw.get("num_heads", 4)
            mem_len = kw.get("mem_len", 1)
            mem_init = kw.get("mem_init", "mean")
            predict_net = DIAMNet(pattern_dim, graph_dim, hidden_dim,
                                  num_heads=num_heads, recurrent_steps=recurrent_steps,
                                  mem_len=mem_len, mem_init=mem_init,
                                  dropout=self.dropout, dropatt=self.dropatt)

        elif predict_type == "CCANet":
            hidden_dim = kw.get("hidden_dim", 64)
            act_func = "relu"
            predict_net = CCANet(pattern_dim, graph_dim, hidden_dim)
        else:
            raise NotImplementedError("Currently, %s is not supported!" % (predict_type))
        return predict_net
