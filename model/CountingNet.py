from model.GraphModel import GraphModel
from model.HUGNN import HUGNN
from model.motifNet import MotifGNN


class Mocha(GraphModel):
    def __init__(self, config, training=True):
        super(Mocha, self).__init__(config)

        # self.ignore_norm = config["rgcn_ignore_norm"]
        self.predict_net_name = config['predict_net']
        # create networks
        # embed the node features and edge features
        p_emb_dim, g_emb_dim, p_e_emb_dim, g_e_emb_dim = self.get_emb_dim()
        # self.pre_g_enc = Embedding(config['init_g_dim'], g_emb_dim)
        self.g_net, g_dim = self.create_graph_net(
            hidden_dim=config["num_g_hid"],
            num_layers=config["graph_num_layers"], num_e_hid=128,
            dropout=self.dropout, model_type=config['graph_net'], gsl=config["GSL"], visualize_only=config['test_only'])

        self.p_net, p_dim = self.create_pattern_net(
            name="pattern", input_dim=p_emb_dim, hidden_dim=config["motif_hidden_dim"],
            num_layers=config["motif_num_layers"],
            dropout=self.dropout, model_type=config['motif_net'])
        # create predict layer
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
        gsl = kwargs.get("gsl", "True")
        out_dim = kwargs.get("out_dim", 64)
        visualize_only = kwargs.get("visualize_only", False)
        net = HUGNN(num_layers=num_layers, input_dim=input_dim, num_g_hid=hidden_dim, num_e_hid=e_hidden_dim,
                    model_type=model_type, out_dim=out_dim, dropout=dropout, gsl=gsl, visualize_only=visualize_only)
        return net, out_dim

    def create_pattern_net(self, input_dim, **kwargs):
        # num_layers, num_g_hid, num_e_hid, out_g_ch, model_type, dropout
        num_layers = kwargs.get("num_layers", 3)
        hidden_dim = kwargs.get("num_g_hid", 128)
        e_hidden_dim = kwargs.get("num_e_hid", 128)
        dropout = kwargs.get("dropout", 0.2)
        model_type = kwargs.get("model_type", "NNGINConcat")
        output_dim = kwargs.get("out_g_ch", 64)
        net = MotifGNN(num_layers=num_layers, num_g_hid=hidden_dim, num_e_hid=e_hidden_dim, dropout=dropout,
                       model_type=model_type, out_g_ch=output_dim)
        return net, output_dim

    def forward(self, motif_x, motif_edge_index, motif_edge_attr, graph):
        pattern_emb = self.p_net(motif_x, motif_edge_index, motif_edge_attr)
        graph_output = self.g_net(graph)
        if self.predict_net_name.startswith("Film") or self.predict_net_name.startswith(
                "CCA") or self.predict_net_name.startswith("Wasserstein"):
            pred, var, filmreg = self.predict_net(pattern_emb, graph_output)
            # distribution, filmreg = self.predict_net(pattern_emb, graph_output)
            # filmreg = (torch.sum(alpha ** 2)) ** 0.5 + (torch.sum(beta ** 2)) ** 0.5
            return pred, var, filmreg
        else:
            pred, var = self.predict_net(pattern_emb, graph_output)
            return pred, var
