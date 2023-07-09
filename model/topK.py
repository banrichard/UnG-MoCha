from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max

from torch_geometric.utils import softmax

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.inits import uniform


def topk(
        score: Tensor,
        x: Tensor,
        ratio: Optional[Union[float, int]],
        batch: Tensor,
        min_score: Optional[float] = None,
        tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all edges in a graph.
        scores_max = scatter_max(score, batch)[0].index_select(0, batch) - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (score > scores_min).nonzero().view(-1)

    elif ratio is not None:
        num_edges = scatter_add(batch.new_ones(score.size(0)), batch, dim=0)
        num_nodes = scatter_add(batch.new_ones(score.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = score.new_full((batch_size * max_num_nodes,), -60000.0)
        dense_x[index] = score
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
            k = torch.min(k, num_nodes)
        else:
            k = (float(ratio) * num_nodes.to(score.dtype)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=score.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    else:
        raise ValueError("At least one of 'min_score' and 'ratio' parameters "
                         "must be specified")

    return perm


def filter_adj(
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        perm: Tensor,
        num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class EdgeTopKPooling(torch.nn.Module):
    r"""This is copied from torch_geometric official version,we modified it into edge-centric but not node-centric.
    Original version can be found at torch_geometric website.
    """

    def __init__(
            self,
            in_channels: int,
            ratio: Union[int, float] = 0.5,
            min_score: Optional[float] = None,
            multiplier: float = 1.,
            nonlinearity: Callable = torch.tanh,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        size = self.in_channels
        uniform(size, self.weight)

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor] = None,
            batch: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:
        """"""

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # self attention
        attn = edge_attr
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        # y = XP/||P|| -> E_A*P/||P||
        score = (attn * self.weight).sum(dim=-1)
        # y = sigmoid(y(idx))
        if self.min_score is None:
            # actually here is tanh
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        perm = topk(score, x, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')
