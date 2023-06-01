import math
import os
import queue
import random
from multiprocessing import Pool

import pandas as pd
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, degree
from collections import defaultdict, OrderedDict
import scipy.sparse as ssp
import numpy as np
import networkx as nx
from tqdm import tqdm

from utils.batch import Batch


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
    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes
    if type(data.num_nodes) is torch.Tensor:
        num_nodes = num_nodes.item()
    edge_attr = data.edge_attr
    for h_ in h:
        subgraphs = []
        for e in edge_index.T:
            nodes_0, edge_index_0, edge_mask_0, z_0 = k_hop_subgraph(
                e[0], h_, edge_index, False, num_nodes, node_label='hop',
                max_nodes_per_hop=max_nodes_per_hop
            )
            nodes_1, edge_index_1, edge_mask_1, z_1 = k_hop_subgraph(
                e[1], h_, edge_index, False, num_nodes, node_label='hop',
                max_nodes_per_hop=max_nodes_per_hop
            )
            nodes_ = [nodes_0[0], nodes_1[0]]
            nodes_ = nodes_ + [item for item in nodes_0 if item not in nodes_]
            nodes_ = nodes_ + [item for item in nodes_1 if item not in nodes_]
            edge_mask_ = torch.logical_or(edge_mask_0, edge_mask_1)
            z_ = []
            for n in nodes_:
                d0 = z_0[n] if n in z_0 else h_ + 1
                d1 = z_1[n] if n in z_1 else h_ + 1
                z_.append([d0, d1])
            z_ = torch.tensor(z_).to(edge_index.device)
            nodes_ = torch.tensor(nodes_).to(edge_index.device)
            edge_index_ = edge_index[:, edge_mask_]
            # relabel nodes
            node_idx = edge_index[1].new_full((num_nodes,), -1)
            node_idx[nodes_] = torch.arange(nodes_.size(0), device=edge_index.device)
            edge_index_ = node_idx[edge_index_]

            x_ = None
            edge_attr_ = None
            pos_ = None
            if x is not None:
                x_ = x[nodes_]
            else:
                x_ = None

            if 'node_type' in data:
                node_type_ = data.node_type[nodes_]

            if data.edge_attr is not None:
                edge_attr_ = edge_attr[edge_mask_]
            if data.pos is not None:
                pos_ = data.pos[nodes_]
            data_ = data.__class__(x_, edge_index_, edge_attr_, None, pos_, z=z_)
            data_.num_nodes = nodes_.shape[0]
            data_.sub_degree = degree(edge_index_[0], num_nodes=nodes_.shape[0])
            data_.original_idx = nodes_

            if 'node_type' in data:
                data_.node_type = node_type_

            subgraphs.append(data_)

        pos_encs = []
        pos_indices = []
        pos_batches = []
        cnt_batch = 0
        for sg in subgraphs:
            # encodes the distance and degree information of nodes within the subgraph
            lsg = sg.num_nodes
            pos_enc = torch.cat((F.one_hot(sg.sub_degree.long(), num_classes=200).view(lsg, -1),
                                 F.one_hot(sg.z.long(), num_classes=100).view(lsg, -1)), dim=-1)
            pos_enc = pos_enc.sum(dim=0)

            # encodes the edge information within the subgraph
            # wrong version, cannot count number of edges of certain type
            # pos_enc = torch.cat((pos_enc, F.one_hot(sg.z[sg.edge_index].transpose(0, 1).reshape(-1, 4), num_classes=100).view(-1,400).sum(dim=0)))
            pos_enc = torch.cat((pos_enc,
                                 F.one_hot((sg.z[remove_self_loops(sg.edge_index)[0]].transpose(0, 1).reshape(-1,
                                                                                                              4)) @ torch.tensor(
                                     [216, 36, 6, 1]), num_classes=1300).sum(dim=0)))
            # pos_encs.append(pos_enc.unsqueeze(0))#.to_sparse())
            pos_index = torch.nonzero(pos_enc)
            pos_encs.append(pos_enc[pos_index].view(-1))
            pos_indices.append(pos_index.view(-1))
            pos_batches.append(torch.LongTensor([cnt_batch for _ in range(pos_index.size()[0])]))
            cnt_batch += 1

        if not hasattr(data, 'pos'):
            new_data = data.__class__(data.x, edge_index, edge_attr, data.y, None, pos_enc=torch.cat(pos_encs, dim=0),
                                      pos_index=torch.cat(pos_indices, dim=0), pos_batch=torch.cat(pos_batches, dim=0))
        elif not hasattr(data, 'name'):
            new_data = data.__class__(data.x, edge_index, edge_attr, data.y, None, pos_enc=torch.cat(pos_encs, dim=0),
                                      pos_index=torch.cat(pos_indices, dim=0), pos_batch=torch.cat(pos_batches, dim=0))
        else:
            new_data = data.__class__(data.x, edge_index, edge_attr, data.y, pos=data.pos,
                                      pos_enc=torch.cat(pos_encs, dim=0), pos_index=torch.cat(pos_indices, dim=0),
                                      pos_batch=torch.cat(pos_batches, dim=0), name=data.name, node_type=data.node_type)
    return new_data


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



def _to_cuda(l):
    """
    put a list of tensor to gpu
    """
    return [t.cuda() for t in l]


def _to_dataloaders(datasets, batch_size=1, shuffle=True):
    """
    create a lists of torch dataloader from datasets
    """

    dataloaders = [DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
                   for dataset in datasets] if isinstance(datasets, list) \
        else [DataLoader(dataset=datasets, batch_size=batch_size, shuffle=shuffle)]
    return dataloaders
