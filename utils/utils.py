import queue
import random

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from collections import defaultdict
import scipy.sparse as ssp
import numpy as np
import networkx as nx


# from batch import Batch


def create_subgraphs(data, h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                     node_label='hop', use_rd=False, subgraph_pretransform=None):
    # Given a PyG graph data, extract an h-hop rooted subgraph for each of its
    # nodes, and combine these node-subgraphs into a new large disconnected graph
    # If given a list of h, will return multiple subgraphs for each node stored in
    # a dict.

    if type(h) == int:
        h = [h]
    assert (isinstance(data, Data))
    # TODO: modify the data obtaining
    edge_index, num_nodes = data.edge_index, data.num_nodes
    new_data_multi_hop = {}
    # traverse the h-hop neighbors
    for h_ in h:
        subgraphs = []
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_, z_ = k_hop_subgraph(
                ind, h_, edge_index, True, num_nodes, node_label=node_label,
                max_nodes_per_hop=max_nodes_per_hop
            )
            x_ = None
            edge_attr_ = None
            pos_ = None

            if 'node_type' in data:
                node_type_ = data.node_type[nodes_]

            if data.edge_attr is not None:
                edge_attr_ = data.edge_attr[edge_mask_]
            if data.pos is not None:
                pos_ = data.pos[nodes_]
            data_ = data.__class__(x_, edge_index_, edge_attr_, None, pos_, z=z_)
            data_.num_nodes = nodes_.shape[0]

            if 'node_type' in data:
                data_.node_type = node_type_

            if subgraph_pretransform is not None:  # for k-gnn
                data_ = subgraph_pretransform(data_)

            subgraphs.append(data_)

        # new_data is treated as a big disconnected graph of the batch of subgraphs
        new_data = Batch.from_data_list(subgraphs)
        new_data.num_nodes = sum(data_.num_nodes for data_ in subgraphs)
        new_data.num_subgraphs = len(subgraphs)

        new_data.original_edge_index = edge_index
        new_data.original_edge_attr = data.edge_attr
        new_data.original_pos = data.pos

        # rename batch, because batch will be used to store node_to_graph assignment
        new_data.node_to_subgraph = new_data.batch
        del new_data.batch

        # create a subgraph_to_graph assignment vector (all zero)
        new_data.subgraph_to_graph = torch.zeros(len(subgraphs), dtype=torch.long)

        # copy remaining graph attributes
        for k, v in data:
            if k not in ['x', 'edge_index', 'edge_attr', 'pos', 'num_nodes', 'batch',
                         'z', 'rd', 'node_type']:
                new_data[k] = v

        if len(h) == 1:
            return new_data
        else:
            new_data_multi_hop[h_] = new_data

    return new_data_multi_hop


# TODO: modify this function to get TOP-K subgraphs of each node
def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target', node_label='hop',
                   max_nodes_per_hop=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    visited = set(subsets[-1].tolist())
    label = defaultdict(list)
    for node in subsets[-1].tolist():
        label[node].append(1)
    if node_label == 'hop':
        hops = [torch.LongTensor([0], device=row.device).flatten()]
    for h in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        new_nodes = col[edge_mask]  # select the 1-hop neighbors
        tmp = []
        for node in new_nodes.tolist():
            if node in visited:
                continue
            tmp.append(node)
            label[node].append(h + 2)
        if len(tmp) == 0:
            break
        # if max_nodes_per_hop is not None:
        #     if max_nodes_per_hop < len(tmp):
        #         tmp = random.sample(tmp, max_nodes_per_hop)
        new_nodes = set(tmp)
        visited = visited.union(new_nodes)
        new_nodes = torch.tensor(list(new_nodes), device=row.device)
        subsets.append(new_nodes)
        if node_label == 'hop':
            hops.append(torch.LongTensor([h + 1] * len(new_nodes), device=row.device))
    subset = torch.cat(subsets)
    inverse_map = torch.tensor(range(subset.shape[0]))
    if node_label == 'hop':
        hop = torch.cat(hops)
    # Add `node_idx` to the beginning of `subset`.
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    z = None
    if node_label == 'hop':
        hop = hop[hop != 0]
        hop = torch.cat([torch.LongTensor([0], device=row.device), hop])
        z = hop.unsqueeze(1)

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, edge_mask, z


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def pygstyle2nx(data) -> nx.Graph:
    nx_graph = nx.Graph()
    # print(nx_graph.number_of_edges())
    edges = []
    graph_data = pd.read_csv("dataset/krogan/krogan_core.txt", header=None, skiprows=1, delimiter=" ")
    for i in range(len(graph_data)):
        edges.append((graph_data.iloc[i, 0], graph_data.iloc[i, 1], {'prob': graph_data.iloc[i, 2]}))

    nx_graph.add_edges_from(edges)
    return nx_graph


def induced_subgraph(graph, src, neighbors, k=1):
    nodes_list = [src]
    Q = queue.Queue()
    Q.put(src)
    depth = 0
    while not Q.empty():
        s = Q.qsize()
        for _ in range(s):
            cur = Q.get()
            for next in graph.neighbors(cur):
                if next in nodes_list:
                    continue
                Q.put(next)
                nodes_list.append(next)
        depth += 1
        if depth >= k:
            break
    subgraph = nx.subgraph(graph, nodes_list).copy()
    # filter the nodes and edges
    remove_node_list = []
    for node in subgraph.nodes():
        if node not in neighbors:
            remove_node_list.append(node)
    subgraph.remove_nodes_from(remove_node_list)

    remove_edge_list = []
    for (u, v) in subgraph.edges():
        if subgraph.edges[u, v]['prob'] < random.random() and (
                u not in subgraph.neighbors(src) or v not in subgraph.neighbors(src)):
            remove_edge_list.append((u, v))
    # refine the subgraph and remove the edges that can't form a connected subgraph with the root node.
    subgraph.remove_edges_from(remove_edge_list)
    # largest_cp = max(nx.connected_components(subgraph))

    return subgraph.copy()


def k_hop_random_walk(dataset, hop=1, walks=10, subs=5):
    graph = pygstyle2nx(dataset[0])
    # Set random seed
    torch.manual_seed(1234)
    candidate_sets = {}
    # Generate 1-hop subgraphs using random walk

    for node in graph.nodes:
        subgraphs = []
        for sub in range(subs):
            # Start random walk from node
            node_idx = torch.tensor([node])
            walk = [node]
            for i in range(walks):  # 10 steps of random walk
                neighbors = list(graph.neighbors(node))
                if len(neighbors) == 0:
                    break
                next_node = torch.tensor([neighbors[torch.randint(len(neighbors), size=(1,))]])
                walk.append(next_node.item())

            # Convert random walk to subgraph
            subgraph_nodes = set(walk)

            neighbors = subgraph_nodes.copy()
            neighbors.remove(node)

            subgraph = induced_subgraph(graph, node, subgraph_nodes)
            edge_index = torch.tensor(list(subgraph.edges)).T.contiguous()
            x = torch.tensor(np.eye(subgraph.number_of_nodes()))

            # subgraph = nx.subgraph_view(subgraph, filter_node=filter_nodes, filter_edge=filter_edges)
            # for neighbor in neighbors:
            #     if graph[node][neighbor]['prob'] >= random.random():
            #         subgraph_edges.append((node, neighbor, graph[node][neighbor]['prob']))
            #         subgraph_prob *= graph[node][neighbor]['prob']
            # adj = nx.to_scipy_sparse_array(subgraph).tocoo()
            # row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            # col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            # edge_index = np.empty((2, subgraph.number_of_edges()))
            # for i in range(len(row) - 1):
            #     for j in range(i + 1, len(col) - 1):
            #         edge_index[i, j] =
            edge_attr = []
            for (u, v) in subgraph.edges:
                edge_attr.append(subgraph.edges[u, v])
            edge_attr = [x['prob'] for x in edge_attr]
            edge_attr = torch.Tensor(edge_attr).reshape(-1, 1).to(torch.float64)
            subgraph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            subgraphs.append(subgraph)

        candidate_sets[node] = subgraphs
    return candidate_sets


def neighbors(fringe, A):
    # Find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        _, out_nei, _ = ssp.find(A[node, :])
        in_nei, _, _ = ssp.find(A[:, node])
        nei = set(out_nei).union(set(in_nei))
        res = res.union(nei)
    return res


class return_prob(object):
    def __init__(self, steps=50):
        self.steps = steps

    def __call__(self, data):
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        adj += ssp.identity(data.num_nodes, dtype='int', format='csr')
        rp = np.empty([data.num_nodes, self.steps])
        inv_deg = ssp.lil_matrix((data.num_nodes, data.num_nodes))
        inv_deg.setdiag(1 / adj.sum(1))
        P = inv_deg * adj
        if self.steps < 5:
            Pi = P
            for i in range(self.steps):
                rp[:, i] = Pi.diagonal()
                Pi = Pi * P
        else:
            inv_sqrt_deg = ssp.lil_matrix((data.num_nodes, data.num_nodes))
            inv_sqrt_deg.setdiag(1 / (np.array(adj.sum(1)) ** 0.5))
            B = inv_sqrt_deg * adj * inv_sqrt_deg
            L, U = eigh(B.todense())
            W = U * U
            Li = L
            for i in range(self.steps):
                rp[:, i] = W.dot(Li)
                Li = Li * L

        data.rp = torch.FloatTensor(rp)

        return data
