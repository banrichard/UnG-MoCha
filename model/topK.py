from typing import Callable, Optional, Tuple, Union, Any

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.utils import dense_to_sparse, softmax, remove_isolated_nodes
from model.MLP import MLP, FC
from model.sparse_softmax import Sparsemax
from utils.graph_operator import data_graph_transform
import torch.nn.functional as f
from torch_geometric.loader import DataLoader
from utils.graph_operator import maximal_component
from dataset_generator import UGDataset


class EdgeScore(torch.nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.mlp = MLP(in_ch=in_channels, hid_ch=128, out_ch=1)

    def forward(self, edge_attr, batch):
        edge_score = self.mlp(edge_attr.view(-1, 2))
        # num_edges = edge_index.size(1)
        edge_score = edge_score.view(-1, 1)
        return edge_score


class TopKEdgePooling(torch.nn.Module):
    r"""This is copied from torch_geometric official version.We modified it into edge-centric but not node-centric.
    Original version can be found at torch_geometric website.
    """

    def __init__(
            self,
            in_channels: int = 2,
            min_score: Optional[float] = None,
            ratio: Union[int, float] = None,
            nonlinearity: Callable = torch.tanh,
            epsilon=1e-6,
            negative_slop=0.2,
            lamb=0.1
    ):
        super().__init__()
        self.min_score = min_score
        self.ratio = ratio
        self.nonlinearity = nonlinearity
        self.epsilon = epsilon
        self.negative_slop = negative_slop
        self.sparse_attention = Sparsemax()
        self.lamb = lamb
        self.in_channels = in_channels
        self.edge_score = EdgeScore(in_channels=in_channels)

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x, edge_index, edge_attr, batch, edge_batch = data.x, data.edge_index, data.edge_attr.view(-1,
                                                                                                   2), data.node_to_subgraph, data.edge_to_subgraph
        device = x.device
        pi = self.edge_score(edge_attr=edge_attr, batch=edge_batch)
        score = self.gumbel_softmax(pi, batch=edge_batch, training=self.training)
        score = score.view(-1)
        perm = topk(score, self.ratio, edge_batch, self.min_score)
        # filter the selected nodes and edges
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]
        edge_batch = edge_batch[perm]
        # unique_nodes = torch.unique(edge_index)
        # x = x.index_select(0, unique_nodes)
        # batch = batch.index_select(0, unique_nodes)
        # Remove isolated nodes should be processed in subgraphs but not the whole concatenated graph:
        # TODO: write a method to process in each subgraph
        x, edge_index, edge_attr, batch = maximal_component(x, edge_attr,
                                                            edge_index, batch,
                                                            edge_batch,
                                                            data.num_subgraphs)
        # update node feature
        # x, batch = filter_nodes(edge_index=edge_index, x=x, batch=batch, edge_batch=edge_batch, perm=perm)
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
    graph = UGDataset()
    loader = DataLoader(graph)
    topk_sample = TopKEdgePooling(ratio=0.5, in_channels=2)
    x, edge_index, edge_attr, batch = topk_sample(graph)
    edge_attr = edge_attr[:, 0].view(-1, 1).expand(-1, 128)
    gnn_for_test = GINConv(nn=torch.nn.Sequential(torch.nn.Linear(128, 256),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(256, 256)),
                           train_eps=True)
    y = gnn_for_test(x=x, edge_index=edge_index)
    unique_nodes = torch.unique(edge_index)
    node_mask = torch.isin(batch, unique_nodes)
    x = x[node_mask]
    batch = batch[node_mask]
    num_nodes = y.size(0)
    print(y)
    # edge_score = EdgeScore(in_channels=1)
    # score = edge_score(edge_attr, edge_batch)
