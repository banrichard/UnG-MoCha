import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv, global_mean_pool, TopKPooling


class SimpleTopK():
    """
    This class is for HUGNN demo: to manually select one potential subgraph rooted at node n.
    """
    def jaccard_distance(self,graph1, graph2):
        set1 = set(graph1.edge_index.numpy().flatten())
        set2 = set(graph2.edge_index.numpy().flatten())

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return 1.0 - intersection / union
    def top_k_scores(self,batch, k):
        num_graphs = batch.num_subgraphs
        scores = []

        for i in range(num_graphs):
            for j in range(i + 1, num_graphs):
                graph1 = batch[i]
                graph2 = batch[j]
                distance = self.jaccard_distance(graph1, graph2)
                scores.append(distance)

        scores = torch.tensor(scores)
        top_k_indices = scores.argsort()[:k]
        top_k_scores = scores[top_k_indices]

        return top_k_scores


class TopK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(TopK, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GraphConv(hidden, hidden, aggr='mean'))
            self.pools.append(TopKPooling(hidden, ratio=0.8))
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv, pool in zip(self.convs, self.pools):
            conv.reset_parameters()
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, (conv, pool) in enumerate(zip(self.convs, self.pools)):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0:
                x, edge_index, _, batch, _ = pool(x, edge_index, batch=batch)
        x = torch.cat(xs, dim=1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

