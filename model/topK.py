from typing import Callable, Optional, Tuple, Union, Any

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_max

from torch_geometric.utils import softmax

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn.inits import uniform
from torch_sparse import SparseTensor

from model.MLP import MLP


# def topk(
#         score: Tensor,
#         batch: Tensor,
#         min_score: Optional[float] = None,
#         ratio: Optional[Union[float, int]] = None,
#         tol: float = 1e-7,
# ) -> Tensor:
#     if min_score is not None:
#         # Make sure that we do not drop all edges in a graph.
#         scores_max = scatter_max(score, batch)[0].index_select(0, batch) - tol
#         scores_min = scores_max.clamp(max=min_score)
#
#         perm = (score > scores_min).nonzero().view(-1)
#     elif ratio is not None:
#         batch_edge = edge_index[0]
#         # num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
#         # problem found here
#         num_edges = scatter_add(edge_index[0].new_ones(score.size(0)), batch_edge, dim=0)
#         # batch_size, max_num_nodes = batch.size(0), int(num_nodes.max())
#         batch_edge_size, max_num_edges = batch_edge.size(0), int(batch_edge.max())
#         # cum_num_nodes = torch.cat(
#         #     [num_nodes.new_zeros(1),
#         #      num_nodes.cumsum(dim=0)[:-1]], dim=0)
#         cum_num_edges = torch.cat([num_edges.new_zeros(1), num_edges.cumsum(dim=0)[:-1]], dim=0)
#         # edge_index = edge_index.T
#         index_edge = torch.arange(batch_edge.size(0), dtype=torch.long, device=batch_edge.device)
#         index_edge = (index_edge - cum_num_edges[batch_edge]) + (batch_edge * max_num_edges)
#         # index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
#         # index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)
#         dense_edge = score.new_full((batch_edge_size * max_num_edges,), 0)
#         # # dense_x = score.new_full((batch_size * max_num_nodes,), -60000.0)
#         dense_edge[index_edge] = score
#         dense_edge = dense_edge.view(batch_edge_size, max_num_edges)
#         # dense_x[index] = score
#         # dense_x = dense_x.view(batch_size, max_num_nodes)
#         _, perm_edge = score.sort(dim=-1, descending=True)
#         # _, perm = dense_x.sort(dim=-1, descending=True)
#         perm_edge = perm_edge + cum_num_edges.view(-1, 1)
#         perm_edge = perm_edge.view(-1)
#         # perm = perm + cum_num_nodes.view(-1, 1)
#         # perm = perm.view(-1)
#
#         if ratio >= 1:
#             # k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
#             # k = torch.min(k, num_nodes)
#             k = batch_edge.new_full((batch_edge.size(0),), int(ratio))
#             k = torch.min(k, batch_edge)
#         else:
#             # k = (float(ratio) * num_nodes.to(score.dtype)).ceil().to(torch.long)
#             k = (float(ratio) * batch_edge.to(score.dtype)).ceil().to(torch.long)
#         # mask = [
#         #     torch.arange(k[i], dtype=torch.long, device=score.device) +
#         #     i * max_num_nodes for i in range(batch_size)
#         # ]
#         edge_mask = [torch.arange(k[i], dtype=torch.long, device=score.device) + i * max_num_edges for i in
#                      range(batch_edge_size)]
#         # mask = torch.cat(mask, dim=0)
#         edge_mask = torch.cat(edge_mask, dim=0)
#         perm_edge = perm_edge[edge_mask]
#
#     else:
#         raise ValueError("At least one of 'min_score' and 'ratio' parameters "
#                          "must be specified")
#
#     return perm_edge


# TODO： try gumble_softmax method to sparse the subgraphs
class TopKEdgePooling(MessagePassing):
    r"""This is copied from torch_geometric official version.We modified it into edge-centric but not node-centric.
    Original version can be found at torch_geometric website.
    """

    def __init__(
            self,
            in_channels: int,
            min_score: Optional[float] = None,
            ratio: Union[int, float] = None,
            nonlinearity: Callable = torch.tanh,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.min_score = min_score
        self.ratio = ratio
        self.nonlinearity = nonlinearity

        self.mlp = MLP(in_channels * 3, in_channels, 1)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        score = self.mlp(torch.concat((x_i, x_j, edge_attr), dim=1))
        return score

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | Any]:
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(x.size(0), x.size(0))).t()
        score = self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x)
        score = score.sum(dim=-1)
        score = self.nonlinearity(score / self.mlp.fc2.weight.norm(p=2, dim=-1))
        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index=edge_index, edge_attr=edge_attr, perm=perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')
