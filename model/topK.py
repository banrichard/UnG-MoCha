import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv, global_mean_pool, TopKPooling


class SimpleTopK():
    """
    This class is for HUGNN demo: to manually select one potential subgraph rooted at node n.
    """

    def __init__(self, candidate_set, k=1):
        self.candidate_set = candidate_set
        self.k = k
    def top_k_subgraphs(self, candidate_set):
        k = self.k
        scores = []

        # Compute a score for each subgraph in the candidate set
        for candidate in candidate_set:
            # Perform some computation or scoring function on the candidate subgraph
            score = self.compute_score(candidate, candidate_set)
            scores.append(score)

        # Sort the candidate subgraphs based on their scores
        sorted_indices = torch.argsort(torch.tensor(scores), descending=True)

        # Select the top-k subgraphs
        top_k_indices = sorted_indices[:k]

        # Return the top-k subgraphs from the candidate set
        top_k_subgraphs = [candidate_set[i] for i in top_k_indices]
        return top_k_subgraphs

    def compute_score(self, subgraph, candidate_set):
        scores = []

        # Compute the Euclidean distance for each subgraph in the candidate set
        for candidate in candidate_set:
            if candidate != subgraph:
                score = self.euclidean_distance(subgraph, candidate)
                scores.append(score)

        # Return the average Euclidean distance as the subgraph's representation score
        score = torch.mean(torch.tensor(scores))
        return score

    def euclidean_distance(self, subgraph1, subgraph2):
        # Extract the 'prob' attribute from the edges of both subgraphs
        prob1 = subgraph1.edge_attr  # Assuming 'prob' is the first attribute
        prob2 = subgraph2.edge_attr

        # Compute the Euclidean distance between the edge probabilities
        distance = torch.norm(prob1 - prob2, p=2)  # Euclidean distance (L2 norm)
        return distance


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


