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


class EdgeScore(torch.nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.mlp = MLP(in_ch=in_channels, hid_ch=in_channels * 2, out_ch=1)

    # TODO: involve batch to calculate the edge score for each subgraph
    def forward(self, edge_attr, batch):
        edge_score = self.mlp(edge_attr)

        # num_edges = edge_index.size(1)
        edge_score = softmax(edge_score, batch).view(-1)
        return edge_score


class Score(MessagePassing):
    def __init__(self, in_channels=128, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.mlp = MLP(in_channels * 3, in_channels, 1)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        score = self.mlp(torch.concat((x_i, x_j, edge_attr), dim=1))
        return score

    def forward(self, x, edge_index, edge_attr):
        score = self.propagate(edge_index=edge_index, edge_attr=edge_attr, x=x)
        return score


def filter_nodes(edge_index, batch, edge_batch):
    pass


class TopKEdgePooling(MessagePassing):
    r"""This is copied from torch_geometric official version.We modified it into edge-centric but not node-centric.
    Original version can be found at torch_geometric website.
    """

    def __init__(
            self,
            in_channels: int = 1,
            min_score: Optional[float] = None,
            ratio: Union[int, float] = None,
            nonlinearity: Callable = torch.tanh,
            epsilon=1e-6,
            negative_slop=0.2,
            lamb=0.1,
            training=True
    ):
        super().__init__()
        self.min_score = min_score
        self.ratio = ratio
        self.nonlinearity = nonlinearity
        self.epsilon = epsilon
        self.negative_slop = negative_slop
        self.sparse_attention = Sparsemax()
        self.lamb = lamb
        self.edge_score = EdgeScore(in_channels=in_channels)
        self.training = training

    def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Optional[Tensor] = None,
            edge_batch: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        edge_score = self.edge_score(edge_attr=edge_attr, batch=edge_batch)
        pi = softmax(edge_score, batch)
        # pi = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # score = torch.sparse_coo_tensor(indices=edge_index, values=score, size=size)
        # score = softmax(score)
        # score = torch.abs(score).sum(dim=-1)
        score = self.gumbel_softmax(pi, batch=batch, training=self.training)
        score = score.view(-1)
        perm = topk(score, self.ratio, batch, self.min_score)
        # update node feature
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]
        edge_batch = edge_batch[perm]
        # num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        # batch = batch[mask]
        # x = x[mask]
        # new_edge_index, new_edge_attr = neighbor_aug(edge_index, edge_attr, num_nodes=score.size(0))

        edge_index, edge_attr = filter_adj(edge_index=edge_index, edge_attr=edge_attr, perm=perm,
                                           num_nodes=score.size(0))

        # row, col = edge_index
        # weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1) + self.lamb * edge_attr[:, 0]
        # non_zero_mask = weights != 0
        # # Filter values and indices using the mask
        # filtered_values = weights[non_zero_mask]
        # filtered_indices = edge_index[:, non_zero_mask]
        # row, col = filtered_indices[0, :], filtered_indices[1, :]
        # edge_attr = self.sparse_attention(filtered_values, row)
        # filter out zero weight edges
        return x, edge_index, edge_attr, batch

    def gumbel_softmax(self, logits, temperature=0.1, batch=None, training=False):
        gumbel_noise = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(gumbel_noise + self.epsilon) + self.epsilon)

        # Add the Gumbel noise to the logits
        logits_gumbel = (logits + gumbel_noise) / temperature if training else logits / temperature

        # Apply softmax to obtain the Gumbel-Softmax distribution
        probs = softmax(logits_gumbel, batch)
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
    edge_batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    # topk_sample = TopKEdgePooling(ratio=0.5)
    # y = topk_sample(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    edge_score = EdgeScore(in_channels=1)
    score = edge_score(edge_attr, edge_batch)
