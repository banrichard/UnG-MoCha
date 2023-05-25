import torch
import torch.nn as nn
import torch.nn.functional as F


def extend_dimensions(old_layer, new_input_dim=-1, new_output_dim=-1, upper=False):
    if isinstance(old_layer, nn.Linear):
        old_output_dim, old_input_dim = old_layer.weight.size()
        if new_input_dim == -1:
            new_input_dim = old_input_dim
        if new_output_dim == -1:
            new_output_dim = old_output_dim
        assert new_input_dim >= old_input_dim and new_output_dim >= old_output_dim

        if new_input_dim != old_input_dim or new_output_dim != old_output_dim:
            use_bias = old_layer.bias is not None
            new_layer = nn.Linear(new_input_dim, new_output_dim, bias=use_bias)
            with torch.no_grad():
                nn.init.zeros_(new_layer.weight)
                if upper:
                    new_layer.weight[:old_output_dim, :old_input_dim].data.copy_(old_layer.weight)
                else:
                    new_layer.weight[-old_output_dim:, -old_input_dim:].data.copy_(old_layer.weight)
                if use_bias:
                    nn.init.zeros_(new_layer.bias)
                    if upper:
                        new_layer.bias[:old_output_dim].data.copy_(old_layer.bias)
                    else:
                        new_layer.bias[-old_output_dim:].data.copy_(old_layer.bias)
        else:
            new_layer = old_layer

    elif isinstance(old_layer, nn.LayerNorm):
        old_input_dim = old_layer.normalized_shape
        if len(old_input_dim) != 1:
            raise NotImplementedError
        old_input_dim = old_input_dim[0]
        assert new_input_dim >= old_input_dim
        if new_input_dim != old_input_dim and old_layer.elementwise_affine:
            new_layer = nn.LayerNorm(new_input_dim, elementwise_affine=True)
            with torch.no_grad():
                nn.init.ones_(new_layer.weight)
                nn.init.zeros_(new_layer.bias)
                if upper:
                    new_layer.weight[:old_input_dim].data.copy_(old_layer.weight)
                    new_layer.bias[:old_input_dim].data.copy_(old_layer.bias)
                else:
                    new_layer.weight[-old_input_dim:].data.copy_(old_layer.weight)
                    new_layer.bias[-old_input_dim:].data.copy_(old_layer.bias)
        else:
            new_layer = old_layer

    return new_layer


class BasePoolPredictNet(nn.Module):
    def __init__(self, pattern_dim, graph_dim, hidden_dim, dropout=0.0):
        super(BasePoolPredictNet, self).__init__()
        self.pattern_dim = pattern_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.p_layer = nn.Linear(pattern_dim, hidden_dim)
        self.g_layer = nn.Linear(graph_dim, hidden_dim)

        self.pred_layer1 = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.pred_layer2 = nn.Linear(self.hidden_dim, 1)

        # init
        for layer in [self.p_layer, self.g_layer, self.pred_layer1]:
            nn.init.normal_(layer.weight, 0.0, 1 / (self.hidden_dim ** 0.5))
            nn.init.zeros_(layer.bias)
        for layer in [self.pred_layer2]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, pattern, pattern_len, graph, graph_len):
        raise NotImplementedError

    def increase_input_size(self, new_pattern_dim, new_graph_dim):
        assert new_pattern_dim >= self.pattern_dim and new_graph_dim >= self.graph_dim
        if new_pattern_dim != self.pattern_dim:
            new_p_layer = extend_dimensions(self.p_layer, new_input_dim=new_pattern_dim, upper=False)
            del self.p_layer
            self.p_layer = new_p_layer
        if new_graph_dim != self.graph_dim:
            new_g_layer = extend_dimensions(self.g_layer, new_input_dim=new_graph_dim, upper=False)
            del self.g_layer
            self.g_layer = new_g_layer
        self.pattern_dim = new_pattern_dim
        self.graph_dim = new_graph_dim


class FilmSumPredictNet(BasePoolPredictNet):
    def __init__(self, pattern_dim, graph_dim, hidden_dim, dropout=0.2):
        super(FilmSumPredictNet, self).__init__(pattern_dim, graph_dim, hidden_dim, dropout)
        self.linear_alpha1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_alpha2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_beta1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_beta2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear_alpha1.weight)
        torch.nn.init.zeros_(self.linear_alpha1.bias)
        torch.nn.init.xavier_uniform_(self.linear_alpha2.weight)
        torch.nn.init.zeros_(self.linear_alpha2.bias)
        torch.nn.init.xavier_uniform_(self.linear_beta1.weight)
        torch.nn.init.zeros_(self.linear_beta1.bias)
        torch.nn.init.xavier_uniform_(self.linear_beta2.weight)
        torch.nn.init.zeros_(self.linear_beta2.bias)

    def forward(self, pattern, graph):
        p = self.drop(self.p_layer(torch.sum(pattern, dim=1, keepdim=True)))
        g = self.drop(self.g_layer(graph))
        _p = p.expand(-1, g.size(1), -1)

        alpha = self.linear_alpha1(g) + self.linear_alpha2(_p)
        beta = self.linear_beta1(g) + self.linear_beta2(_p)
        g = (alpha + 1) * g + beta
        p = p.squeeze()
        g = torch.sum(g, dim=1)
        y = self.pred_layer1(
            torch.cat([p, g, g - p, g * p], dim=1))  # W ^T * FCL(x ‖ y ‖ x − y ‖ x \dot y) + b.
        y = self.act(y)  # relu
        y = self.pred_layer2(y)
        # return y
        filmreg = (torch.sum(alpha ** 2)) ** 0.5 + (torch.sum(beta ** 2)) ** 0.5
        return y, filmreg


class PatternReadout(nn.Module):
    def __init__(self, input_channels, output_channels, dropout):
        super(PatternReadout, self).__init__()
        self.W_readout_p = torch.nn.Parameter(torch.FloatTensor(input_channels, output_channels))
        self.input_c = input_channels
        self.output_c = output_channels
        self.bias_readout_p = torch.nn.Parameter(torch.FloatTensor(1, output_channels))
        self.act = nn.ReLU()
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_readout_p)
        torch.nn.init.zeros_(self.bias_readout_p)

    def forward(self, edge_attr):
        # mean aggregator
        mean = edge_attr.mean(dim=1)
        result = torch.matmul(mean, self.W_readout_p) + self.bias_readout_p
        result = F.dropout(result, p=self.dropout, training=self.training)
        result = self.act(result)
        return result


class GraphReadout(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(GraphReadout, self).__init__()

        self.W_gamma = torch.nn.Parameter(torch.FloatTensor(input_channels, input_channels))
        self.U_gamma = torch.nn.Parameter(torch.FloatTensor(input_channels, input_channels))
        self.bias_gamma = torch.nn.Parameter(torch.FloatTensor(1, input_channels))
        self.W_beta = torch.nn.Parameter(torch.FloatTensor(input_channels, input_channels))
        self.U_beta = torch.nn.Parameter(torch.FloatTensor(input_channels, input_channels))
        self.bias_beta = torch.nn.Parameter(torch.FloatTensor(1, input_channels))

        self.W_readout_g = torch.nn.Parameter(torch.FloatTensor(input_channels, output_channels))
        self.bias_readout_g = torch.nn.Parameter(torch.FloatTensor(1, output_channels))

        self.input_c = input_channels
        self.output_c = output_channels
        self.act = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_gamma)
        torch.nn.init.xavier_uniform_(self.U_gamma)
        torch.nn.init.xavier_uniform_(self.bias_gama)
        torch.nn.init.xavier_uniform_(self.W_beta)
        torch.nn.init.xavier_uniform_(self.U_beta)
        torch.nn.init.xavier_uniform_(self.bias_beta)
        torch.nn.init.xavier_uniform_(self.W_readout_g)
        torch.nn.init.xavier_uniform_(self.bias_readout_g)

    def forward(self, edge_attr, pattern_embedding):
        # mean
        # formula(9)(10)(8)
        one = torch.ones(len(edge_attr), len(edge_attr[0]), self.input_c)

        ################################################
        ####################sum#########################
        ################################################
        gamma = self.act(torch.matmul(edge_attr, self.W_gamma) + one * torch.matmul(pattern_embedding,
                                                                                    self.U_gamma) + one * self.bias_gamma)
        beta = self.act(torch.matmul(edge_attr, self.W_beta) + one * torch.matmul(pattern_embedding,
                                                                                  self.U_beta) + one * self.bias_beta)
        _edge_attr = (gamma + one) * edge_attr + beta
        mean = _edge_attr.mean(dim=1)
        result = torch.matmul(mean, self.W_readout_g) + self.bias_readout_g
        result = self.act(result)
        return result, gamma, beta


class PredictNet(nn.Module):
    def __init__(self, length, dropout):
        super(PredictNet, self).__init__()
        self.W_counter = torch.nn.Parameter(torch.FloatTensor(length, length))
        self.bias_counter = torch.nn.Parameter(torch.FloatTensor(1, length))
        self.reset_parameters()
        self.length = length
        self.act = nn.ReLU()
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_counter)
        torch.nn.init.xavier_uniform_(self.bias_counter)

    def forward(self, pattern_embedding, graph_embedding):
        p_e = pattern_embedding.reshape(self.length, 1)
        g_e = graph_embedding.reshape(1, self.length)
        Relu = F.relu(torch.matmul(g_e, self.W_counter) + self.bias_counter)
        # result=F.leaky_relu(torch.matmul(leakyR,p_e))
        result = torch.matmul(Relu, p_e)
        result = torch.squeeze(result)
        result = F.dropout(result, p=self.dropout, training=self.training)
        result = self.act(result)
        return result
