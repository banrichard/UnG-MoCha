from typing import Callable, Optional, Tuple, Union, Any

import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
import torch.nn.functional as f
from torch_geometric.utils import dense_to_sparse, softmax
from torch_scatter import scatter_add
from torch_sparse import SparseTensor
from torch.sparse import Tensor as st
from model.MLP import MLP
from model.sparse_softmax import Sparsemax


def batch_filter(x, batch, edge_index, edge_attr):
    unique_elements, num_nodes = batch.unique(return_counts=True)
    mask = num_nodes > 2
    idx = unique_elements[mask]
    # orginal_x = x
    node_mask = torch.isin(batch, idx)
    batch = batch[node_mask]
    x = x[node_mask]
    # edge_index, edge_attr = filter_adj(edge_index, edge_attr, node_mask, orginal_x.size(0))

    return x, batch, edge_index, edge_attr


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
            epsilon=1e-6,
            negative_slop=0.2,
            lamb=0.1
    ):
        super().__init__()

        self.in_channels = in_channels
        self.min_score = min_score
        self.ratio = ratio
        self.nonlinearity = nonlinearity
        self.epsilon = epsilon
        self.negative_slop = negative_slop
        self.mlp = MLP(in_channels * 3, in_channels, 1)
        self.att = torch.nn.Parameter(torch.Tensor(1, in_channels * 2))
        self.sparse_attention = Sparsemax()
        self.lamb = lamb
        torch.nn.init.xavier_uniform(self.att)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        score = self.mlp(torch.concat((x_i, x_j, edge_attr), dim=1))
        return score

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        score = self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x)
        score = torch.abs(score).sum(dim=-1)
        perm = topk(score, self.ratio, batch, self.min_score)
        # update node feature
        x = x[perm]
        batch = batch[perm]
        # num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        # batch = batch[mask]
        # x = x[mask]
        # new_edge_index, new_edge_attr = neighbor_aug(edge_index, edge_attr, num_nodes=score.size(0))

        edge_index, edge_attr = filter_adj(edge_index=edge_index, edge_attr=edge_attr, perm=perm,
                                           num_nodes=score.size(0))

        row, col = edge_index
        weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1) + self.lamb * edge_attr[:, 0]
        non_zero_mask = weights != 0
        # Filter values and indices using the mask
        filtered_values = weights[non_zero_mask]
        filtered_indices = edge_index[:, non_zero_mask]
        row, col = filtered_indices[0, :], filtered_indices[1, :]
        edge_attr = self.sparse_attention(filtered_values, row)
        # filter out zero weight edges
        return x, edge_index, edge_attr, batch

    def gumbel_softmax(self, logits, temperature=1.0, hard=False):
        gumbel_noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise + self.epsilon) + self.epsilon)

        # Add the Gumbel noise to the logits
        logits_gumbel = (logits + gumbel_noise) / temperature

        # Apply softmax to obtain the Gumbel-Softmax distribution
        probs = f.softmax(logits_gumbel, dim=-1)

        if hard:
            # Hard Gumbel-Softmax by sampling the argmax
            _, argmax = torch.max(probs, dim=-1)
            one_hot = torch.zeros_like(logits).scatter_(-1, argmax.unsqueeze(-1), 1.0)
            probs_hard = (one_hot - probs).detach() + probs

            return probs_hard

        return probs

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return f'{self.__class__.__name__}({self.in_channels}, {ratio})'


if __name__ == "__main__":
    x = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]], dtype=torch.float)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    batch_filter(x, batch, edge_index, edge_attr)
