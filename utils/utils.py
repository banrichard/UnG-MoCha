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
from .batch import Batch


def nodes_to_subgraphs(data):
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    candidate_sets = data.sets
    subgraphs = []
    for k in candidate_sets.keys():
        subgraphs.append(candidate_sets[k])
    # get each node's subgraph representation
    new_data = Batch.from_data_list(subgraphs)
    # create a subgraph_to_graph assignment vector (all zero)
    new_data.num_subgraphs = len(subgraphs)

    new_data.original_edge_index = data.edge_index
    new_data.original_edge_attr = data.edge_attr
    # rename batch, because batch will be used to store node_to_graph assignment
    new_data.node_to_subgraph = new_data.batch
    del new_data.batch
    new_data.subgraph_to_graph = torch.zeros(len(subgraphs), dtype=torch.long)
    # new_datas.append(new_data)
    return new_data


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
    edge_index, num_nodes, edge_attr = data.edge_index, data.num_nodes, data.edge_attr
    new_data_multi_hop = {}
    # traverse the h-hop neighbors
    for h_ in h:
        subgraphs = []
        for ind in range(num_nodes):
            nodes_, edge_index_, edge_mask_ = new_k_hop_rw(ind, h_, edge_index, edge_attr, subs=1)
            # node_set = set()
            # subgraph_edge_index = candidate_set.edge_index
            # for i in range(len(subgraph_edge_index)):
            #     for j in range(len(subgraph_edge_index[0])):
            #         node_set.add(subgraph_edge_index[i][j])

            # nodes_ = torch.tensor(list(node_set))
            edge_attr_ = data.edge_attr[edge_mask_]
            data_ = Data(edge_index=edge_index_, edge_attr=edge_attr_)
            subgraphs.append(data_)

        # new_data is treated as a big disconnected graph of the batch of subgraphs
        new_data = Batch.from_data_list(subgraphs)
        # the sum of nodes in each subgraph
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


def pygstyle2nx(filepath) -> nx.Graph:
    nx_graph = nx.Graph()
    # print(nx_graph.number_of_edges())
    edges = []
    graph_data = pd.read_csv(filepath, header=None, skiprows=1, delimiter=" ")
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


def k_hop_random_walk(hop=1, walks=10, subs=5):
    graph = pygstyle2nx("dataset/krogan/krogan_core.txt")
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

            edge_attr = []
            for (u, v) in subgraph.edges:
                edge_attr.append(subgraph.edges[u, v])
            edge_attr = [x['prob'] for x in edge_attr]
            edge_attr = torch.Tensor(edge_attr).reshape(-1, 1).to(torch.float64)
            subgraph = Data(edge_index=edge_index, edge_attr=edge_attr)
            subgraphs.append(subgraph)

        candidate_sets[node] = subgraphs[0]

    return candidate_sets


def subgraph_padding(candidate_set):
    max_nodes = max([subgraph.num_nodes for subgraph in candidate_set])
    max_edges = max([subgraph.num_edges for subgraph in candidate_set])
    padded_subgraphs = []
    masks = []

    for key in candidate_set:
        mask, padded_subgraph = subgraph_process(max_edges, max_nodes, key)
        padded_subgraphs.append(padded_subgraph)
        masks.append(mask)

    # Create a batch from padded subgraphs
    # batch = Batch.from_data_list(padded_subgraphs)

    # Stack the masks into a tensor
    mask_tensor = torch.stack(masks)
    return padded_subgraphs, mask_tensor


def subgraph_process(max_edges, max_nodes, subgraph):
    num_nodes_to_pad = max_nodes - subgraph.num_nodes
    num_edges_to_pad = max_edges - subgraph.num_edges
    # Pad node features, edge indices, and edge attributes
    padded_x = torch.cat([subgraph.x, torch.zeros((num_nodes_to_pad, subgraph.x.shape[1]))], dim=0)
    padded_edge_index = torch.cat([subgraph.edge_index,
                                   torch.zeros((2, num_edges_to_pad), dtype=torch.long)], dim=1)
    padded_edge_attr = torch.cat([subgraph.edge_attr,
                                  torch.zeros((num_edges_to_pad, subgraph.edge_attr.shape[1]))], dim=0)
    # Create mask
    mask = torch.cat([torch.ones(subgraph.num_nodes, dtype=torch.uint8),
                      torch.zeros(num_nodes_to_pad, dtype=torch.uint8)], dim=0)
    padded_subgraph = Data(x=padded_x, edge_index=padded_edge_index, edge_attr=padded_edge_attr)
    return mask, padded_subgraph


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


def new_k_hop_rw(node_idx, num_hops, edge_index, edge_attr,
                 num_nodes=None,
                 max_nodes_per_hop=None, walks=10, subs=5):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    col_mask = col.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    subsets = [torch.tensor([node_idx], device=row.device).flatten()]
    visited = set(subsets[-1].tolist())

    label = defaultdict(list)
    for node in subsets[-1].tolist():
        label[node].append(1)
    for h in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        col_mask.fill_(False)
        col_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)

        # select the neighbors from root node as the source node
        new_nodes = col[edge_mask]

        # select the neighbors from other nodes to root node
        col_mask = torch.index_select(col_mask, 0, col)
        edge_mask[[col_mask == True]] = True
        new_nodes = torch.cat((new_nodes, row[edge_mask]))
        new_nodes = new_nodes[~np.isin(new_nodes, subsets[0])]
        tmp = []
        walk = 0
        for walk in range(walks):
            if walk > walks:
                break
            next_node = new_nodes[random.randint(0, len(new_nodes) - 1)]
            pos = edge_index.size(1) - 1
            for i in range(edge_index.size(1)):
                if edge_mask[i] == False:
                    continue
                if edge_index[:, i].equal(torch.Tensor([subsets[0], next_node]).int()) or edge_index[:, i].equal(
                        torch.Tensor([next_node, subsets[0]]).int()):
                    pos = i
                    break
                # edge_attr[edge_index[0] == subsets[0] and ]
            if edge_attr[pos] < random.random():
                continue
            tmp.append(int(next_node))
            walk += 1

        if len(tmp) == 0:
            break
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(tmp):
                tmp = random.sample(tmp, max_nodes_per_hop)
        new_nodes = set(tmp)
        visited = visited.union(new_nodes)
        new_nodes = torch.tensor(list(new_nodes), device=row.device)
        subsets.append(new_nodes)
    subset = torch.cat(subsets)
    # Add `node_idx` to the beginning of `subset`.
    subset = subset[subset != node_idx]
    subset = torch.cat([torch.tensor([node_idx], device=row.device), subset])

    node_mask.fill_(False)
    node_mask[subset] = True

    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    return subset, edge_index, edge_mask
