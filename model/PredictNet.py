import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.MLP import MLP
from utils.utils import batch_convert_len_to_mask, gather_indices_by_lens


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
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.pred_layer1 = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        # self.pred_layer2 = nn.Linear(self.hidden_dim, 1)
        self.ln = nn.LayerNorm(hidden_dim)
        self.pred_mean = nn.Linear(self.hidden_dim, 1)
        self.pred_var = nn.Linear(self.hidden_dim, 1)
        # init
        for layer in [self.p_layer, self.g_layer, self.pred_layer1]:
            nn.init.normal_(layer.weight, 0.0, 1 / (self.hidden_dim ** 0.5))
            nn.init.zeros_(layer.bias)
        for layer in [self.pred_mean, self.pred_var]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, pattern, graph):
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
        self.epsilon = 1e-8
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
        p = self.drop(self.p_layer(pattern))
        g = self.drop(self.g_layer(graph))
        # _p = p.expand(-1, g.size(1), -1)

        alpha = self.linear_alpha1(g) + self.linear_alpha2(p)
        beta = self.linear_beta1(g) + self.linear_beta2(p)
        g = (alpha + 1) * g + beta
        # p = p.squeeze()
        # g = torch.sum(g, dim=1)
        y = self.pred_layer1(
            torch.cat([p, g, g - p, g * p], dim=1))  # W ^T * FCL(x ‖ y ‖ x − y ‖ x \dot y) + b.
        y = self.act(y)  # relu
        y = self.ln(y)
        y_var = y
        y = self.pred_mean(y)
        y = F.relu(y)
        var = self.pred_var(y_var)
        var = F.relu(var)
        # distribution = torch.distributions.Normal(loc=y + self.epsilon, scale=torch.sqrt(var) + self.epsilon)
        # return y
        filmreg = (torch.sum(alpha ** 2)) ** 0.5 + (torch.sum(beta ** 2)) ** 0.5
        return y, var, filmreg


class DIAMNet(nn.Module):
    def __init__(self, pattern_dim, graph_dim, hidden_dim, recurrent_steps=1, num_heads=4, mem_len=4, mem_init="mean",
                 dropout=0.2, dropatt=0.2):
        super(DIAMNet, self).__init__()
        self.pattern_dim = pattern_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.mem_len = mem_len
        self.mem_init = mem_init
        self.recurrent_steps = recurrent_steps

        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.p_layer = nn.Linear(pattern_dim, hidden_dim)
        self.g_layer = nn.Linear(graph_dim, hidden_dim)
        if mem_init.endswith("attn"):
            self.m_layer = MultiHeadAttn(
                query_dim=hidden_dim, key_dim=graph_dim, value_dim=graph_dim,
                hidden_dim=hidden_dim, num_heads=num_heads,
                dropatt=dropatt, act_func="softmax")
        elif mem_init.endswith("lstm"):
            self.m_layer = nn.LSTM(graph_dim, hidden_dim, batch_first=True)
        else:
            self.m_layer = self.g_layer
        self.p_attn = GatedMultiHeadAttn(
            query_dim=hidden_dim, key_dim=pattern_dim, value_dim=pattern_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            pre_lnorm=True,
            dropatt=dropatt, act_func="softmax")
        self.g_attn = GatedMultiHeadAttn(
            query_dim=hidden_dim, key_dim=graph_dim, value_dim=graph_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            pre_lnorm=True,
            dropatt=dropatt, act_func="softmax")
        self.m_attn = GatedMultiHeadAttn(
            query_dim=hidden_dim, key_dim=hidden_dim, value_dim=hidden_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            pre_lnorm=True,
            dropatt=dropatt, act_func="softmax")

        self.pred_layer1 = nn.Linear(self.mem_len * self.hidden_dim, self.hidden_dim)
        self.pred_layer_mean = nn.Linear(self.hidden_dim, 1)
        self.pred_layer_var = nn.Linear(self.hidden_dim, 1)
        # init
        scale = 1 / (self.hidden_dim ** 0.5)
        for layer in [self.p_layer, self.g_layer, self.pred_layer1]:
            nn.init.normal_(layer.weight, 0.0, scale)
            nn.init.zeros_(layer.bias)
        for layer in [self.pred_layer_mean, self.pred_layer_var]:
            nn.init.kaiming_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        if isinstance(self.m_layer, nn.LSTM):
            for layer_weights in self.m_layer._all_weights:
                for w in layer_weights:
                    if "weight" in w:
                        weight = getattr(self.m_layer, w)
                        nn.init.orthogonal_(weight)
                    elif "bias" in w:
                        bias = getattr(self.m_layer, w)
                        if bias is not None:
                            nn.init.zeros_(bias)
        elif isinstance(self.m_layer, nn.Linear):
            nn.init.normal_(layer.weight, 0.0, scale)
            nn.init.zeros_(layer.bias)

    def forward(self, pattern, graph):
        bsz = 1
        pattern = pattern.unsqueeze(0)
        graph = graph.unsqueeze(0)
        p_len, g_len = pattern.size(1), graph.size(1)
        # p_mask = batch_convert_len_to_mask(pattern_len) if p_len == pattern_len.max() else None
        # g_mask = batch_convert_len_to_mask(graph_len) if g_len == graph_len.max() else None
        # pattern_len  = (n,1) 0: batch_size 1:number of nodes in the motif
        pattern_len = torch.tensor(pattern.size(0), dtype=torch.int32).view(-1, 1)
        graph_len = torch.tensor(graph.size(0), dtype=torch.int32).view(-1, 1)
        p_mask = batch_convert_len_to_mask(pattern_len).to(pattern.device)
        g_mask = batch_convert_len_to_mask(graph_len).to(graph.device)

        p, g = pattern, graph
        if g_mask is not None:
            mem = list()
            mem_mask = list()
            # for idx in gather_indices_by_lens(graph_len):
            m, mk = init_mem(g, g_mask, mem_len=self.mem_len, mem_init=self.mem_init, post_proj=self.m_layer)
            mem.append(m)
            mem_mask.append(mk)
            mem = torch.cat(mem, dim=0)
            mem_mask = torch.cat(mem_mask, dim=0)
        for i in range(self.recurrent_steps):
            mem = self.p_attn(mem, p, p, p_mask)
            mem = self.g_attn(mem, g, g, g_mask)

        mem = mem.view(1, -1)
        y = self.pred_layer1(mem)
        y = self.act(y)
        mean = self.pred_layer_mean(y)
        var = self.pred_layer_var(y)
        return mean, var

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

        if self.mem_init.endswith("attn"):
            self.m_layer.increase_input_size(self.hidden_dim, new_graph_dim, new_graph_dim)
        else:
            new_m_layer = extend_dimensions(self.m_layer, new_input_dim=new_graph_dim, upper=False)
            del self.m_layer
            self.m_layer = new_m_layer
        self.p_attn.increase_input_size(self.hidden_dim, new_pattern_dim, new_pattern_dim)
        self.g_attn.increase_input_size(self.hidden_dim, new_graph_dim, new_graph_dim)

        self.pattern_dim = new_pattern_dim
        self.graph_dim = new_graph_dim


def init_mem(x, x_mask=None, mem_len=1, mem_init="mean", **kw):
    assert mem_init in ["mean", "sum", "max", "attn", "lstm",
                        "circular_mean", "circular_sum", "circular_max", "circular_attn", "circular_lstm"]
    pre_proj = kw.get("pre_proj", None)
    post_proj = kw.get("post_proj", None)
    if pre_proj:
        x = pre_proj(x)

    bsz, seq_len, hidden_dim = x.size()
    if seq_len < mem_len:
        mem = torch.cat([x, torch.zeros((mem_len - seq_len, hidden_dim), device=x.device, dtype=x.dtype)], dim=1)
        if x_mask is not None:
            mem_mask = torch.cat(
                [x_mask, torch.zeros((mem_len - seq_len), device=x_mask.device, dtype=x_mask.dtype)], dim=1)
        else:
            mem_mask = None
    elif seq_len == mem_len:
        mem, mem_mask = x, x_mask
    else:
        if mem_init.startswith("circular"):
            pad_len = math.ceil((seq_len + 1) / 2) - 1
            x = F.pad(x.transpose(1, 2), pad=(0, pad_len), mode="circular").transpose(1, 2)
            if x_mask is not None:
                x_mask = F.pad(x_mask.unsqueeze(1), pad=(0, pad_len), mode="circular").squeeze(1)
            seq_len += pad_len
        stride = seq_len // mem_len
        kernel_size = seq_len - (mem_len - 1) * stride
        if mem_init.endswith("mean"):
            mem = F.avg_pool1d(x.transpose(1, 2), kernel_size=kernel_size, stride=stride).transpose(1, 2)
        elif mem_init.endswith("max"):
            mem = F.max_pool1d(x.transpose(1, 2), kernel_size=kernel_size, stride=stride).transpose(1, 2)
        elif mem_init.endswith("sum"):
            mem = F.avg_pool1d(x.transpose(1, 2), kernel_size=kernel_size, stride=stride).transpose(1, 2) * kernel_size
        elif mem_init.endswith("attn"):
            # split and self attention
            mem = list()
            attn = kw.get("attn", None)
            hidden_dim = attn.query_dim
            h = torch.ones((1, hidden_dim), dtype=x.dtype, device=x.device, requires_grad=False).mul_(
                1 / (hidden_dim ** 0.5))
            for i in range(0, seq_len - kernel_size + 1, stride):
                j = i + kernel_size
                m = x[:, i:j]
                mk = x_mask[:, i:j] if x_mask is not None else None
                if attn:
                    h = attn(h, m, m, attn_mask=mk)
                else:
                    m = m.unsqueeze(2)
                    h = get_multi_head_attn_vec(h, m, m, attn_mask=mk, act_layer=nn.Softmax(dim=-1))
                mem.append(h)
            mem = torch.cat(mem, dim=1)
        elif mem_init.endswith("lstm"):
            mem = list()
            lstm = kw["lstm"]
            hx = None
            for i in range(0, seq_len - kernel_size + 1, stride):
                j = i + kernel_size
                m = x[:, i:j]
                _, hx = lstm(m, hx)
                mem.append(hx[0].view(1, -1))
            mem = torch.cat(mem, dim=1)
        if x_mask is not None:
            # print(x_mask.size())
            mem_mask = F.max_pool1d(x_mask.float().unsqueeze(1), kernel_size=kernel_size, stride=stride).squeeze(
                1).byte()
            # mem_mask = F.max_pool1d(x_mask.float(), kernel_size=kernel_size, stride=stride).squeeze(1).byte()
        else:
            mem_mask = None
    if post_proj:
        mem = post_proj(mem)
    return mem, mem_mask


class GatedMultiHeadAttn(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_heads,
                 dropatt=0.2, act_func="softmax", add_zero_attn=False,
                 pre_lnorm=False, post_lnorm=False):
        super(GatedMultiHeadAttn, self).__init__()
        assert hidden_dim % num_heads == 0

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropatt = nn.Dropout(dropatt)

        head_dim = hidden_dim // num_heads

        self.q_net = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k_net = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v_net = nn.Linear(value_dim, hidden_dim, bias=False)
        self.o_net = nn.Linear(hidden_dim, query_dim, bias=False)
        self.g_net = nn.Linear(2 * query_dim, query_dim, bias=True)

        self.act = nn.Softmax()
        self.add_zero_attn = add_zero_attn
        self.pre_lnorm = pre_lnorm
        self.post_lnorm = post_lnorm

        if pre_lnorm:
            self.q_layer_norm = nn.LayerNorm(query_dim)
            self.k_layer_norm = nn.LayerNorm(key_dim)
            self.v_layer_norm = nn.LayerNorm(value_dim)
        if post_lnorm:
            self.o_layer_norm = nn.LayerNorm(query_dim)

        # init
        scale = 1 / (head_dim ** 0.5)
        for m in [self.q_net, self.k_net, self.v_net, self.o_net]:
            nn.init.normal_(m.weight, 0.0, scale)
        # when new data comes, it prefers to output 1 so that the gate is 1
        nn.init.normal_(self.g_net.weight, 0.0, scale)
        nn.init.ones_(self.g_net.bias)

    def forward(self, query, key, value, attn_mask=None):
        ##### multihead attention
        # [bsz x head_len x num_heads x head_dim]
        bsz = query.size(0)
        if self.add_zero_attn:
            key = torch.cat([key,
                             torch.zeros((bsz, 1) + key.size()[2:], dtype=key.dtype, device=key.device)], dim=1)
            value = torch.cat([value,
                               torch.zeros((bsz, 1) + value.size()[2:], dtype=value.dtype, device=value.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.ones((bsz, 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)

        qlen, klen, vlen = query.size(1), key.size(1), value.size(1)

        if self.pre_lnorm:
            ##### layer normalization
            query = self.q_layer_norm(query)
            key = self.k_layer_norm(key)
            value = self.v_layer_norm(value)

        # linear projection
        head_q = self.q_net(query).view(bsz, qlen, self.num_heads, self.hidden_dim // self.num_heads)
        head_k = self.k_net(key).view(bsz, klen, self.num_heads, self.hidden_dim // self.num_heads)
        head_v = self.v_net(value).view(bsz, vlen, self.num_heads, self.hidden_dim // self.num_heads)

        # multi head attention
        attn_vec = get_multi_head_attn_vec(
            head_q=head_q, head_k=head_k, head_v=head_v,
            attn_mask=attn_mask, act_layer=self.act, dropatt=self.dropatt)

        ##### linear projection
        attn_out = self.o_net(attn_vec)

        ##### gate
        gate = F.sigmoid(self.g_net(torch.cat([query, attn_out], dim=2)))
        attn_out = gate * query + (1 - gate) * attn_out

        if self.post_lnorm:
            attn_out = self.o_layer_norm(attn_out)

        return attn_out


class MultiHeadAttn(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_heads,
                 dropatt=0.0, act_func="softmax", add_zero_attn=False,
                 pre_lnorm=False, post_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        assert hidden_dim % num_heads == 0
        assert act_func in ["softmax", "sigmoid"]

        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropatt = nn.Dropout(dropatt)

        head_dim = hidden_dim // num_heads

        self.q_net = nn.Linear(query_dim, hidden_dim, bias=False)
        self.k_net = nn.Linear(key_dim, hidden_dim, bias=False)
        self.v_net = nn.Linear(value_dim, hidden_dim, bias=False)
        self.o_net = nn.Linear(hidden_dim, query_dim, bias=False)

        self.act = nn.Softmax()
        self.add_zero_attn = add_zero_attn
        self.pre_lnorm = pre_lnorm
        self.post_lnorm = post_lnorm

        if pre_lnorm:
            self.q_layer_norm = nn.LayerNorm(query_dim)
            self.k_layer_norm = nn.LayerNorm(key_dim)
            self.v_layer_norm = nn.LayerNorm(value_dim)
        if post_lnorm:
            self.o_layer_norm = nn.LayerNorm(query_dim)

        # init
        scale = 1 / (head_dim ** 0.5)
        for m in [self.q_net, self.k_net, self.v_net, self.o_net]:
            nn.init.normal_(m.weight, 0.0, scale)

    def forward(self, query, key, value, attn_mask=None):
        ##### multihead attention
        # [bsz x hlen x num_heads x head_dim]
        bsz = query.size(0)

        if self.add_zero_attn:
            key = torch.cat([key,
                             torch.zeros((bsz, 1) + key.size()[2:], dtype=key.dtype, device=key.device)], dim=1)
            value = torch.cat([value,
                               torch.zeros((bsz, 1) + value.size()[2:], dtype=value.dtype, device=value.device)], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.ones((bsz, 1), dtype=attn_mask.dtype, device=attn_mask.device)], dim=1)

        qlen, klen, vlen = query.size(1), key.size(1), value.size(1)

        if self.pre_lnorm:
            ##### layer normalization
            query = self.q_layer_norm(query)
            key = self.k_layer_norm(key)
            value = self.v_layer_norm(value)

        # linear projection
        head_q = self.q_net(query).view(bsz, qlen, self.num_heads, self.hidden_dim // self.num_heads)
        head_k = self.k_net(key).view(bsz, klen, self.num_heads, self.hidden_dim // self.num_heads)
        head_v = self.v_net(value).view(bsz, vlen, self.num_heads, self.hidden_dim // self.num_heads)

        # multi head attention
        attn_vec = get_multi_head_attn_vec(
            head_q=head_q, head_k=head_k, head_v=head_v,
            attn_mask=attn_mask, act_layer=self.act, dropatt=self.dropatt)

        ##### linear projection
        attn_out = self.o_net(attn_vec)

        if self.post_lnorm:
            attn_out = self.o_layer_norm(attn_out)

        return attn_out


def get_multi_head_attn_vec(head_q, head_k, head_v, attn_mask=None, act_layer=None, dropatt=None):
    _INF = -1e30
    bsz, qlen, num_heads, head_dim = head_q.size()
    # Q = [batch_size * qlen * num_heads * head_dim]
    # K_T = [batch_size * klen * head_dim * num_heads]
    scale = 1 / (head_dim ** 0.5)

    # [bsz x qlen x klen x num_heads]
    attn_score = torch.einsum("bind,bjnd->bijn", (head_q, head_k))
    # attn_score = torch.matmul(head_q, head_k.T)
    attn_score.mul_(scale)
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_score.masked_fill_((attn_mask == 0).unsqueeze(1).unsqueeze(-1), _INF)
        elif attn_mask.dim() == 3:
            attn_score.masked_fill_((attn_mask == 0).unsqueeze(-1), _INF)

    # [bsz x qlen x klen x num_heads]
    if act_layer is not None:
        attn_score = act_layer(attn_score)
    if dropatt is not None:
        attn_score = dropatt(attn_score)

    # [bsz x qlen x klen x num_heads] x [bsz x klen x num_heads x head_dim] -> [bsz x qlen x num_heads x head_dim]
    attn_vec = torch.einsum("bijn,bjnd->bind", (attn_score, head_v))
    attn_vec = attn_vec.contiguous().view(bsz, qlen, -1)
    return attn_vec


class BaseAttnPredictNet(nn.Module):
    def __init__(self, pattern_dim, graph_dim, hidden_dim, num_heads=4, recurrent_steps=1, dropout=0.0, dropatt=0.0):
        super(BaseAttnPredictNet, self).__init__()
        self.pattern_dim = pattern_dim
        self.grpah_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.recurrent_steps = recurrent_steps

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.p_layer = nn.Linear(pattern_dim, hidden_dim)
        self.g_layer = nn.Linear(graph_dim, hidden_dim)

        self.p_attn = GatedMultiHeadAttn(
            query_dim=graph_dim, key_dim=pattern_dim, value_dim=pattern_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            pre_lnorm=True,
            dropatt=dropatt, act_func="softmax")
        self.g_attn = GatedMultiHeadAttn(
            query_dim=graph_dim, key_dim=graph_dim, value_dim=graph_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            pre_lnorm=True,
            dropatt=dropatt, act_func="softmax")

        self.pred_layer1 = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.pred_layer_mean = nn.Linear(self.hidden_dim, 1)
        self.pred_layer_var = nn.Linear(self.hidden_dim, 1)

        # init
        for layer in [self.p_layer, self.g_layer, self.pred_layer1]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.pred_layer_mean, self.pred_layer_var]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, pattern, graph):
        raise NotImplementedError


class MeanAttnPredictNet(BaseAttnPredictNet):
    def __init__(self, pattern_dim, graph_dim, hidden_dim, num_heads=4, recurrent_steps=1, dropout=0.0, dropatt=0.0):
        super(MeanAttnPredictNet, self).__init__(pattern_dim, graph_dim, hidden_dim, num_heads, recurrent_steps,
                                                 dropout, dropatt)

    def forward(self, pattern, graph):
        bsz = 1
        pattern = pattern.unsqueeze(0)
        graph = graph.unsqueeze(0)
        p_len, g_len = pattern.size(1), graph.size(1)
        # p_mask = batch_convert_len_to_mask(pattern_len) if p_len == pattern_len.max() else None
        # g_mask = batch_convert_len_to_mask(graph_len) if g_len == graph_len.max() else None
        # pattern_len  = (n,1) 0: batch_size 1:number of nodes in the motif
        pattern_len = torch.tensor(pattern.size(0), dtype=torch.int32).view(-1, 1)
        graph_len = torch.tensor(graph.size(0), dtype=torch.int32).view(-1, 1)
        p_mask = batch_convert_len_to_mask(pattern_len).to(pattern.device)
        g_mask = batch_convert_len_to_mask(graph_len).to(graph.device)

        p, g = pattern, graph
        for i in range(self.recurrent_steps):
            g = self.p_attn(g, p, p, p_mask)
            g = self.g_attn(g, g, g, g_mask)

        p = self.drop(self.p_layer(torch.mean(p, dim=1, keepdim=True)))
        g = self.drop(self.g_layer(g))

        p = p.squeeze(1)
        g = torch.mean(g, dim=1)
        y = self.pred_layer1(torch.cat([p, g, g - p, g * p], dim=1))
        y = self.act(y)
        mean = self.pred_layer_mean(y)
        var = self.pred_layer_var(y)
        return mean, var


class MeanPredictNet(BasePoolPredictNet):
    def __init__(self, pattern_dim, graph_dim, hidden_dim, dropout=0.0):
        super(MeanPredictNet, self).__init__(pattern_dim, graph_dim, hidden_dim, dropout)

    def forward(self, pattern, graph):
        pattern = pattern.unsqueeze(0)
        graph = graph.unsqueeze(0)
        p = self.drop(self.p_layer(torch.mean(pattern, dim=1, keepdim=True)))
        g = self.drop(self.g_layer(graph))
        p = p.squeeze(1)
        g = torch.mean(g, dim=1)
        y = self.pred_layer1(torch.cat([p, g, g - p, g * p], dim=1))
        y = self.act(y)
        mean = self.pred_mean(y)
        var = self.pred_var(y)
        return mean, var


class CCANet(torch.nn.Module):
    def __init__(self, pattern_dim, graph_dim, hidden_dim, dropout=0.5, lam=0.1) -> None:
        super(CCANet, self).__init__()
        self.pattern_dim = pattern_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lam = lam
        self.graph_layer = nn.Linear(graph_dim, hidden_dim)
        self.motif_layer = nn.Linear(pattern_dim, hidden_dim)
        self.pred_layer1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        # self.pred_layer2 = nn.Linear(self.hidden_dim, 1)
        self.ln = nn.LayerNorm(hidden_dim)
        self.pred_mean = nn.Linear(self.hidden_dim, 1)
        self.pred_var = nn.Linear(self.hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.reset_parameter()

    def reset_parameter(self):
        for layer in [self.graph_layer, self.motif_layer, self.pred_layer1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.pred_mean, self.pred_var]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, graph, motif):
        graph_cca = self.ln(self.graph_layer(graph))
        motif_cca = self.ln(self.motif_layer(motif))
        c = torch.mm(graph_cca.T, motif_cca)
        c1 = torch.mm(graph_cca.T, graph_cca)
        c2 = torch.mm(motif_cca.T, motif_cca)
        loss_inv = -torch.diagonal(c).sum()
        iden = torch.eye(c.shape[0]).to(c.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()
        y = self.pred_layer1(torch.cat([graph_cca, motif_cca], dim=1))
        y = self.act(y)
        mean = self.pred_mean(y)
        mean = F.relu(mean)
        var = self.pred_var(y)
        var = F.relu(var)
        cca_reg = loss_inv + self.lam * (loss_dec1 + loss_dec2)
        return mean, var, cca_reg
