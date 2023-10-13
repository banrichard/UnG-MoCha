from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import softmax
from torch_geometric.utils import to_networkx

from dataset_generator import UGDataset
from model.MLP import MLP
from utils.graph_operator import maximal_component
from utils.utils import visualization


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
            lamb=0.1,
            visualize_only=False
    ):
        super().__init__()
        self.min_score = min_score
        self.ratio = ratio
        self.nonlinearity = nonlinearity
        self.epsilon = epsilon
        self.negative_slop = negative_slop
        self.lamb = lamb
        self.in_channels = in_channels
        self.edge_score = EdgeScore(in_channels=in_channels)
        self.visualize_only = visualize_only

    def forward(self, data: Data) -> tuple[
        Tensor, Tensor, Tensor, Tensor]:
        if torch.cuda.is_available():
            x, edge_index, edge_attr, batch, edge_batch = data.x.cuda(), data.edge_index.cuda(), data.edge_attr.view(-1,
                                                                                                                     2).cuda(), \
                data.batch.cuda(), \
                data.edge_batch.cuda()
        else:
            x, edge_index, edge_attr, batch, edge_batch = data.x, data.edge_index, data.edge_attr.view(-1,
                                                                                                       2), data.batch, data.edge_batch
        device = x.device
        score = self.edge_score(edge_attr=edge_attr, batch=edge_batch)
        # get score on every subgraph
        score = softmax(score, edge_batch)
        score = score.view(-1)
        perm = topk(x=score, ratio=self.ratio, batch=edge_batch, min_score=self.min_score)
        # filter the selected nodes and edges, we already have the mask matrix
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]
        edge_batch = edge_batch[perm]
        # Remove isolated nodes should be processed in subgraphs but not the whole concatenated graph:
        # TODO: write a method to process in each subgraph
        x, edge_index, edge_attr, batch = maximal_component(x, edge_index, edge_attr, batch)
        # update node feature
        # x, batch = filter_nodes(edge_index=edge_index, x=x, batch=batch, edge_batch=edge_batch, perm=perm)

        if self.visualize_only:
            tmp_g = to_networkx(
                Data(x=x[batch == 0], edge_index=edge_index[:, edge_batch == 0],
                     edge_attr=edge_attr[edge_batch == 0, 0].cpu().numpy()), to_undirected=True,
                remove_self_loops=True, edge_attrs=['edge_attr'])

            visualization(tmp_g, "gsl")
            exit(0)
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
    graph = UGDataset(root="../dataset")
    edge_batch = graph.edge_batch
    loader = DataLoader(graph, batch_size=graph.len())
    data = next(iter(loader))
    data.edge_batch = edge_batch
    topk_sample = TopKEdgePooling(ratio=0.5, in_channels=2)
    x, edge_index, edge_attr, batch = topk_sample(data)
    edge_attr = edge_attr[:, 0].view(-1, 1)
    gnn_for_test = GINConv(nn=torch.nn.Sequential(torch.nn.Linear(128, 128),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(128, 64)),
                           train_eps=True)
    y = gnn_for_test(x=x, edge_index=edge_index)
    num_nodes = y.size(0)
    print(y)
    # edge_score = EdgeScore(in_channels=1)
    # score = edge_score(edge_attr, edge_batch)
