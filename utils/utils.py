import math
import os
import queue
import random
from multiprocessing import Pool

import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from collections import defaultdict, OrderedDict
import scipy.sparse as ssp
import numpy as np
import networkx as nx
from tqdm import tqdm

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


def load_graph(filepath) -> nx.Graph:
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
    graph = load_graph("dataset/krogan/krogan_core.txt")
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
            # pos = edge_index.size(1) - 1
            # for i in range(edge_index.size(1)):
            #     if edge_mask[i] == False:
            #         continue
            #     if edge_index[:, i].equal(torch.Tensor([subsets[0], next_node]).int()) or edge_index[:, i].equal(
            #             torch.Tensor([next_node, subsets[0]]).int()):
            #         pos = i
            #         break
            #     # edge_attr[edge_index[0] == subsets[0] and ]
            # if edge_attr[pos] < random.random():
            #     continue
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


def get_enc_len(x, base):
    l = 0
    while x:
        l += 1
        x = x // base
    return l


def int2onehot(x, len_x, base=10):
    if isinstance(x, (int, list)):
        x = np.array(x)
    x_shape = x.shape
    x = x.reshape(-1)
    one_hot = np.zeros((len_x * base, x.shape[0]), dtype=np.float32)
    x = x % (base ** len_x)
    idx = one_hot.shape[0] - base
    while np.any(x):
        x, y = x // base, x % base
        cond = y.reshape(1, -1) == np.arange(0, base, dtype=y.dtype).reshape(base, 1)
        one_hot[idx:idx + base] = np.where(cond, 1.0, 0.0)
        idx -= base
    while idx >= 0:
        one_hot[idx] = 1.0
        idx -= base
    one_hot = one_hot.transpose(1, 0).reshape(*x_shape, len_x * base)
    return one_hot


def anneal_fn(fn, t, T, lambda0=0.0, lambda1=1.0):
    if not fn or fn == "none":
        return lambda1
    elif fn == "logistic":
        K = 8 / T
        return float(lambda0 + (lambda1 - lambda0) / (1 + np.exp(-K * (t - T / 2))))
    elif fn == "linear":
        return float(lambda0 + (lambda1 - lambda0) * t / T)
    elif fn == "cosine":
        return float(lambda0 + (lambda1 - lambda0) * (1 - math.cos(math.pi * t / T)) / 2)
    elif fn.startswith("cyclical"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R * T, lambda0, lambda1)
        else:
            return anneal_fn(fn.split("_", 1)[1], t - R * T, R * T, lambda1, lambda0)
    elif fn.startswith("anneal"):
        R = 0.5
        t = t % T
        if t <= R * T:
            return anneal_fn(fn.split("_", 1)[1], t, R * T, lambda0, lambda1)
        else:
            return lambda1
    else:
        raise NotImplementedError


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, min_percent=0.0):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(min_percent,
                   float(num_training_steps - current_step) / float(max(1.0, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def load_data(graph_dir, pattern_dir, num_workers=4):
    patterns = read_patterns_from_dir(pattern_dir, num_workers=num_workers)
    graphs = read_graphs_from_dir(graph_dir, num_workers=num_workers)

    train_data, dev_data, test_data = list(), list(), list()
    train_count = 0
    dev_count = 0
    train_length = 5000
    dev_length = 500
    num = 0
    for p, pattern in patterns.items():
        if p in graphs:
            for g, graph in graphs[p].items():
                x = dict()
                x["id"] = ("%s-%s" % (p, g))
                x["pattern"] = pattern
                x["graph"] = graph
                g_idx = int(g.rsplit("_", 1)[-1])

                if num % 100 == 0 and train_count < train_length:
                    train_data.append(x)
                    train_count += 1
                elif num % 100 == 1 and dev_count < dev_length:
                    dev_data.append(x)
                    dev_count += 1
                else:
                    test_data.append(x)
                num += 1
        elif len(graphs) == 1 and "raw" in graphs.keys():
            for g, graph in graphs["raw"].items():
                x = dict()
                x["id"] = ("%s-%s" % (p, g))
                x["pattern"] = pattern
                x["graph"] = graph
                g_idx = int(g.rsplit("_", 1)[-1])
                if g_idx % 3 == 0:
                    dev_data.append(x)
                elif g_idx % 3 == 1:
                    test_data.append(x)
                else:
                    train_data.append(x)
    return OrderedDict({"train": train_data, "dev": dev_data, "test": test_data})


def read_graphs_from_dir(dirpath, num_workers=4):
    graphs = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir,))))
        pool.close()

        for subdir, x in tqdm(results):
            x = x.get()
            graphs[os.path.basename(subdir)] = x
    return graphs


def read_patterns_from_dir(dirpath, num_workers=4):
    patterns = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir,))))
        pool.close()

        for subdir, x in tqdm(results):
            x = x.get()
            patterns.update(x)
            # patterns[os.path.basename(subdir)] = x
    return patterns


def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):
    bsz = graph_sizes.size(0)
    dim, dtype, device = batched_graph_feats.size(-1), batched_graph_feats.dtype, batched_graph_feats.device

    min_size, max_size = graph_sizes.min(), graph_sizes.max()
    mask = torch.ones((bsz, max_size), dtype=torch.uint8, device=device, requires_grad=False)

    if min_size == max_size:
        return batched_graph_feats.view(bsz, max_size, -1), mask
    else:
        graph_sizes_list = graph_sizes.view(-1).tolist()
        unbatched_graph_feats = list(torch.split(batched_graph_feats, graph_sizes_list, dim=0))
        for i, l in enumerate(graph_sizes_list):
            if l == max_size:
                continue
            elif l > max_size:
                unbatched_graph_feats[i] = unbatched_graph_feats[i][:max_size]
            else:
                mask[i, l:].fill_(0)
                zeros = torch.zeros((max_size - l, dim), dtype=dtype, device=device, requires_grad=False)
                unbatched_graph_feats[i] = torch.cat([unbatched_graph_feats[i], zeros], dim=0)
        return torch.stack(unbatched_graph_feats, dim=0), mask
