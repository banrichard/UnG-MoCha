import math
import os
import queue
import random
from multiprocessing import Pool

import pandas as pd
import torch
import torch.nn.functional as F
import scipy.stats as stats

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, degree
from collections import defaultdict, OrderedDict
import scipy.sparse as ssp
import numpy as np
import networkx as nx
from torch_sparse import spspmm, coalesce
from tqdm import tqdm

from .batch import Batch
import matplotlib.pyplot as plt


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


def data_split_cv(all_sets, num_fold=5, seed=1):
    """
    only support nocumulative learning currently
    """
    random.seed(seed)
    all_fold_train_sets = []
    all_fold_val_sets = []
    for key in sorted(all_sets.keys()):
        random.shuffle(all_sets[key])

    for i in range(num_fold):
        train_sets = []
        val_sets = []
        for key in sorted(all_sets.keys()):
            # generate the key-th set for the i-th fold
            num_instances = int(len(all_sets[key]))
            num_fold_instances = num_instances / num_fold
            start = int(i * num_fold_instances)
            end = num_instances if i == num_fold - 1 else int(i * num_fold_instances + num_fold_instances)
            val_sets.append(all_sets[key][start: end])
            train_sets += (all_sets[key][: start] + all_sets[key][end:])
        all_fold_train_sets.append(train_sets)
        all_fold_val_sets.append(val_sets)
    return all_fold_train_sets, all_fold_val_sets


def print_eval_res(all_eval_res, print_details=True):
    total_loss, total_l1 = 0.0, 0.0
    all_errors = []
    for i, (res, loss, l1, elapse_time) in enumerate(
            all_eval_res):  # (res, loss, loss_var, l1, l1_var, elapse_time)
        print(
            "Evaluation result of {}-th Eval set: Loss= {:.4f}, Loss_var = {:.4f},Avg. L1 Loss= {:.4f}, Avg. L1 var Loss={:.4f} Avg. Pred. Time= {:.9f}(s)"
            .format(i, loss, l1 / len(res), elapse_time / len(res)))
        errors = [(output - card) for card, output in res]
        # errors_var = [(output_var - var) for output_var, var in res_var]
        outputs = [output for _, output in res]
        # cards = [card for card, _ in res]
        # outputs_var = [output_var for _, output_var in res_var]
        # vars = [var for var, _ in res_var]
        # avg_outputs = np.average(outputs)
        # std_outputs = np.std(outputs)
        get_prediction_statistics(errors)
        # get_prediction_statistics(errors_var)
        all_errors += errors
        # all_errors_var += errors_var
        total_loss += loss
        # total_loss_var += loss_var
        total_l1 += l1
        # total_l1_var += l1_var
        if print_details:
            for (card, output) in res:
                print("Card : {:.4f}, Pred {:.4f}, Diff = {:.4f}".format(card, output, output - card))
            # for (var, output_var) in res_var:
            #     print("Var : {:.4f}, Pred {:.4f}, Diff = {:.4f}".format(var, output_var, output_var - var))
            # for (card, output, var, output_var) in (cards, outputs, vars, outputs_var):
            #     print("Card : {:.4f}, Pred {:.4f}, Diff = {:.4f},Var : {:.4f}, Pred_var {:.4f}, Diff_var = {:.4f},"
            #           .format(card, output, output - card, var, output_var, output_var - var))
            # print("Average Estimation:{:.4f}, standard deviation:{:.4f}".format(avg_outputs, std_outputs))
    print("Evaluation result of Eval dataset: Total Loss= {:.4f}, Total L1 Loss= {:.4f}".format(total_loss, total_l1))
    error_median = get_prediction_statistics(all_errors)
    return error_median


def get_prediction_statistics(errors: list):
    lower, upper = np.quantile(errors, 0.25), np.quantile(errors, 0.75)
    print("<" * 80)
    print("Predict Result Profile of {} Queries:".format(len(errors)))
    print("Min/Max: {:.4f} / {:.4f}".format(np.min(errors), np.max(errors)))
    print("Mean: {:.4f}".format(np.mean(errors)))
    print("Median: {:.4f}".format(np.median(errors)))
    print("25%/75% Quantiles: {:.4f} / {:.4f}".format(lower, upper))
    print(">" * 80)
    error_median = abs(upper - lower)
    return error_median


def batch_convert_len_to_mask(batch_lens, max_seq_len=-1):
    # batch_lens [n,1]
    if max_seq_len == -1:
        max_seq_len = max(batch_lens)
    mask = torch.ones((len(batch_lens), max_seq_len), dtype=torch.uint8, device=batch_lens[0].device,
                      requires_grad=False)
    for i, l in enumerate(batch_lens):
        mask[i, l:].fill_(0)
    return mask


def gather_indices_by_lens(lens):
    result = list()
    i, j = 0, 1
    max_j = len(lens)
    indices = np.arange(0, max_j)
    while j < max_j:
        if lens[i] != lens[j]:
            result.append(indices[i:j])
            i = j
        j += 1
    if i != j:
        result.append(indices[i:j])
    return result


def val_to_distribution(mean, var) -> torch.Tensor:
    sigma = torch.sqrt(var)
    norm = stats.norm(loc=mean, scale=sigma)
    x = np.linspace(mean - 3 * sigma, mean + 3 * sigma, 128)
    y = torch.Tensor(norm.pdf(x).astype(float).tolist())
    return y


def neighbor_aug(edge_index, edge_attr, num_nodes):
    n = num_nodes
    fill = 1e16
    value = edge_index.new_full((edge_index.size(1),), fill, dtype=torch.float)

    index, value = spspmm(edge_index, value, edge_index, value, n, n, n, True)

    edge_index = torch.cat([edge_index, index], dim=1)
    value = value.view(-1, *[1 for _ in range(edge_attr.dim() - 1)])
    value = value.expand(-1, *list(edge_attr.size())[1:])
    edge_attr = torch.cat([edge_attr, value], dim=0)
    edge_index, edge_attr = coalesce(edge_index, edge_attr, n, n, op='min')
    edge_attr[edge_attr >= fill] = 0

    return edge_index, edge_attr


def load_graph(filepath, emb=None) -> nx.Graph:
    nx_graph = nx.Graph()
    edges = []
    graph_data = pd.read_csv(filepath, header=None, skiprows=1, delimiter=" ")
    for i in range(len(graph_data)):
        edges.append((graph_data.iloc[i, 0], graph_data.iloc[i, 1], {'edge_attr': graph_data.iloc[i, 2]}))
    nx_graph.add_edges_from(edges)
    if emb is not None:
        node_fea = np.loadtxt(emb, delimiter=" ")[:, 1:]
    else:
        node_fea = np.zeros((nx_graph.number_of_nodes(), 128))
    for i in range(nx_graph.number_of_nodes()):
        nx_graph.add_node(i, x=node_fea[i])

    return nx_graph


def k_hop_induced_subgraph_edge(graph, edge, k=1) -> nx.Graph:
    node_list = []
    edge_list = []
    node_u = edge[0]
    node_v = edge[1]
    node_list.append(node_u)
    node_list.append(node_v)
    for neighbor in graph.neighbors(node_u):
        node_list.append(neighbor)
        edge_list.append((node_u, neighbor))
    for neighbor in graph.neighbors(node_v):
        node_list.append(neighbor)
        edge_list.append((node_v, neighbor))
    node_list = list(set(node_list))
    edge_list = [(u, v, {"edge_attr": graph.edges[u, v]["edge_attr"]})
                 for (u, v) in edge_list]
    subgraph = nx.subgraph(graph, node_list).copy()
    remove_edge_list = [edge for edge in subgraph.edges(data=True) if edge not in edge_list]
    subgraph.remove_edges_from(remove_edge_list)
    # visualization(subgraph, "original")
    return subgraph


def visualization(graph: nx.Graph, name):
    from matplotlib import font_manager
    font_manager.fontManager.addfont("./LinLibertine_R.ttf")
    plt.rcParams.update({
        'figure.figsize': (5, 5),
        'font.family': "Linux Libertine",
        'font.size': 16
    })
    options = {
        "node_color": "#0079FF",
        "node_size": 150,
        "edge_cmap": plt.cm.Blues,
        "with_labels": False,
        "font_size": 16,
        "width": 0.5
    }
    if name == "gsl":
        pos = nx.spring_layout(graph)
    else:
        pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos=pos, **options)
    edge_labels = nx.get_edge_attributes(graph, "edge_attr")
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, font_size=16, font_family="Linux Libertine")
    plt.savefig(name + ".pdf", bbox_inches='tight')
    nx.set_node_attributes(graph,0,"x")
    nx.write_gexf(graph, name + ".gexf")
    plt.show()
