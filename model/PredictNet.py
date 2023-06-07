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
        y = self.pred_layer2(y)
        # return y
        filmreg = (torch.sum(alpha ** 2)) ** 0.5 + (torch.sum(beta ** 2)) ** 0.5
        return y, filmreg


class DIAMNet(nn.Module):
    def __init__(self, pattern_dim, graph_dim, hidden_dim, recurrent_steps=1, num_heads=4, mem_len=4, mem_init="mean",
                 dropout=0.0, dropatt=0.0):
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

        self.pred_layer1 = nn.Linear(self.mem_len * self.hidden_dim + 4, self.hidden_dim)
        self.pred_layer2 = nn.Linear(self.hidden_dim + 4, 1)

        # init
        scale = 1 / (self.hidden_dim ** 0.5)
        for layer in [self.p_layer, self.g_layer, self.pred_layer1]:
            nn.init.normal_(layer.weight, 0.0, scale)
            nn.init.zeros_(layer.bias)
        for layer in [self.pred_layer2]:
            nn.init.zeros_(layer.weight)
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

    def forward(self, pattern, pattern_len, graph, graph_len):
        bsz = pattern_len.size(0)
        p_len, g_len = pattern.size(1), graph.size(1)
        plf, glf = pattern_len.float(), graph_len.float()
        inv_plf, inv_glf = 1.0 / plf, 1.0 / glf
        # p_mask = batch_convert_len_to_mask(pattern_len) if p_len == pattern_len.max() else None
        # g_mask = batch_convert_len_to_mask(graph_len) if g_len == graph_len.max() else None
        p_mask = batch_convert_len_to_mask(pattern_len)
        g_mask = batch_convert_len_to_mask(graph_len)

        p, g = pattern, graph
        if g_mask is not None:
            mem = list()
            mem_mask = list()
            for idx in gather_indices_by_lens(graph_len):
                if self.mem_init.endswith("attn"):
                    m, mk = init_mem(g[idx, :graph_len[idx[0]]], g_mask[idx, :graph_len[idx[0]]],
                                     mem_len=self.mem_len, mem_init=self.mem_init, attn=self.m_layer)
                elif self.mem_init.endswith("lstm"):
                    m, mk = init_mem(g[idx, :graph_len[idx[0]]], g_mask[idx, :graph_len[idx[0]]],
                                     mem_len=self.mem_len, mem_init=self.mem_init, lstm=self.m_layer)
                else:
                    m, mk = init_mem(g[idx, :graph_len[idx[0]]], g_mask[idx, :graph_len[idx[0]]],
                                     mem_len=self.mem_len, mem_init=self.mem_init, post_proj=self.m_layer)
                mem.append(m)
                mem_mask.append(mk)
            mem = torch.cat(mem, dim=0)
            mem_mask = torch.cat(mem_mask, dim=0)
        for i in range(self.recurrent_steps):
            mem = self.p_attn(mem, p, p, p_mask)
            mem = self.g_attn(mem, g, g, g_mask)

        mem = mem.view(bsz, -1)
        y = self.pred_layer1(torch.cat([mem, plf, glf, inv_plf, inv_glf], dim=1))
        y = self.act(y)
        y = self.pred_layer2(torch.cat([y, plf, glf, inv_plf, inv_glf], dim=1))

        return y

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


class GatedMultiHeadAttn(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, num_heads,
                 dropatt=0.0, act_func="softmax", add_zero_attn=False,
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
    scale = 1 / (head_dim ** 0.5)

    # [bsz x qlen x klen x num_heads]
    attn_score = torch.einsum("bind,bjnd->bijn", (head_q, head_k))
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
