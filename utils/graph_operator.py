import os
import queue
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx
import torch_geometric.transforms as T
from utils.batch import Batch


# from dataloader import DataLoader

def node_reorder(query, nodes_list, edges_list):
    idx_dict = {}
    node_cnt = 0
    for v in nodes_list:
        idx_dict[v] = node_cnt
        node_cnt += 1
    nodes_list = [(idx_dict[v], {"labels": query.nodes[v]["labels"]})
                  for v in nodes_list]
    edges_list = [(idx_dict[u], idx_dict[v], {"edge_attr": query.edges[u, v]["edge_attr"]})
                  for (u, v) in edges_list]
    sample = nx.Graph()
    sample.add_nodes_from(nodes_list)
    sample.add_edges_from(edges_list)
    return sample


def data_graph_transform(data_dir, dataset, dataset_name, batch_path=None, gsl=True):
    graph = load_graph(os.path.join(data_dir, dataset, dataset_name),
                       emb=batch_path)
    candidate_sets = {}
    # for node in range(graph.number_of_nodes()):
    #     subgraph = k_hop_induced_subgraph(graph, node)
    #     candidate_sets[node] = random_walk_on_subgraph(subgraph, node)
    cnt = 0
    for edge in graph.edges(data=True):
        subgraph = k_hop_induced_subgraph_edge(graph, edge)
        if gsl:
            candidate_sets[cnt] = subgraph
        else:
            candidate_sets[cnt] = random_walk_on_subgraph_edge(subgraph, edge)
        cnt += 1
    batch = create_batch(graph, candidate_sets, edge_base=True)
    return batch


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
    return subgraph


def k_hop_induced_subgraph(graph, src, k=1):
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
    edges_list = graph.subgraph(nodes_list).edges()
    subgraph = node_reorder(graph, nodes_list, edges_list)

    return subgraph


def candidate_filter(candidate_set):
    final_candidate = None
    prob = 0
    for candidate in candidate_set:
        cur_prob = 1
        for e in candidate.edges(data=True):
            cur_prob *= e[2]['edge_attr']
        if cur_prob > prob:
            final_candidate = candidate
            prob = cur_prob
    return final_candidate


def random_walk_on_subgraph(subgraph: nx.Graph, node, walks=20, subs=5):
    # Set random seed
    random.seed(1)
    candidate_set = []
    # Generate 1-hop subgraphs using random walk
    for s in range(subs):
        node_list = [node]
        for i in range(walks):
            neighbors = list(subgraph.neighbors(node))
            if len(neighbors) == 0:
                break
            next_node = neighbors[random.randint(0, len(neighbors) - 1)]
            node_list.append(next_node)
        node_list = set(node_list)
        tmp_graph = subgraph.subgraph(list(node_list)).copy()
        remove_edge_list = []
        for (u, v) in tmp_graph.edges():
            if tmp_graph.edges[u, v]['edge_attr'] < random.random():
                remove_edge_list.append((u, v))
        tmp_graph.remove_edges_from(remove_edge_list)
        remove_node_list = [node for node in tmp_graph.nodes() if tmp_graph.degree(node) == 0]
        tmp_graph.remove_nodes_from(remove_node_list)
        candidate_set.append(tmp_graph)
    subgraph_with_highest_probability = candidate_filter(candidate_set)
    return subgraph_with_highest_probability


def random_walk_on_subgraph_edge(subgraph: nx.Graph, edge, walks=20, subs=5) -> nx.Graph:
    random.seed(1)
    candidate_set = []
    for s in range(subs):
        # read the two nodes on the root edge
        node1 = edge[0]
        node2 = edge[1]
        node_list = [node1, node2]
        for i in range(walks):
            neighbors = set(subgraph.neighbors(node1)).union(set(subgraph.neighbors(node2)))
            neighbors = list(neighbors)
            if len(neighbors) == 0:
                break
            next_node = neighbors[random.randint(0, len(neighbors) - 1)]
            node_list.append(next_node)
        node_list = set(node_list)
        tmp_graph = subgraph.subgraph(list(node_list)).copy()
        remove_edge_list = []
        for (u, v) in tmp_graph.edges():
            if tmp_graph.edges[u, v]['edge_attr'] < random.random():
                remove_edge_list.append((u, v))
        tmp_graph.remove_edges_from(remove_edge_list)
        remove_node_list = [node for node in tmp_graph.nodes() if tmp_graph.degree(node) == 0]
        tmp_graph.remove_nodes_from(remove_node_list)
        candidate_set.append(tmp_graph)
    subgraph_with_highest_probability = candidate_filter(candidate_set)
    return subgraph_with_highest_probability


def create_batch(graph: nx.Graph, candidate_sets: dict, edge_base=True):
    """
    :param graph: the original graph (the node features are pre-embedded by Node2Vec)
    :param candidate_sets: the candidate subgraph(s) of each node
    :return: a Batch contains subgraph representation
    """
    # convert networkx graph into pyg style
    pyg_graph = torch_geometric.utils.from_networkx(graph, group_edge_attrs=all)
    # init the edge count dict: get keys from pyg_graph edge index
    edge_count = {tuple(edge.numpy()): 0 for edge in pyg_graph.edge_index.t()}
    x = torch.zeros(len(graph.nodes), 1, dtype=torch.float32)
    pyg_graph.x = x
    pyg_subgraphs = []
    if edge_base:
        for cnt in range(graph.number_of_edges()):
            subgraph = candidate_sets[cnt]
            pyg_subgraph = torch_geometric.utils.from_networkx(subgraph)
            if pyg_subgraph.num_nodes == 0:
                continue
            org_attr = pyg_subgraph.edge_attr.clone().detach()
            edge_attr = torch.zeros_like(pyg_subgraph.edge_index.t()).to(torch.float32)
            for i in range(pyg_subgraph.edge_index.t().size(0)):
                edge = pyg_subgraph.edge_index.t()[i]
                if tuple(edge.numpy()) in edge_count.keys():
                    edge_count[tuple(edge.numpy())] = edge_count.get(tuple(edge.numpy())) + 1
                    edge_attr[i] = torch.tensor((org_attr[i], torch.tensor(edge_count[tuple(edge.numpy())])))
            pyg_subgraph.edge_attr = edge_attr
            pyg_subgraphs.append(pyg_subgraph)
    else:
        for node in range(graph.number_of_nodes()):
            subgraph = candidate_sets[node]
            pyg_subgraph = torch_geometric.utils.from_networkx(subgraph)
            if pyg_subgraph.num_nodes == 0:
                continue
            pyg_subgraphs.append(pyg_subgraph)

    pyg_batch = Batch.from_data_list(pyg_subgraphs)

    pyg_batch.x = pyg_batch.x.to(torch.float32)
    pyg_batch.edge_attr = pyg_batch.edge_attr.to(torch.float32)
    pyg_batch.edge_index = torch.LongTensor(pyg_batch.edge_index)
    pyg_batch.num_nodes = sum(data_.num_nodes for data_ in pyg_subgraphs)
    pyg_batch.num_edges = sum(data_.num_edges for data_ in pyg_subgraphs)
    pyg_batch.num_subgraphs = len(pyg_subgraphs)

    pyg_batch.original_x = pyg_graph.x
    pyg_batch.original_edge_index = pyg_graph.edge_index
    pyg_batch.original_edge_attr = pyg_graph.edge_attr
    pyg_batch.node_to_subgraph = pyg_batch.batch
    pyg_batch.edge_to_subgraph = pyg_batch.edge_batch
    del pyg_batch.batch
    del pyg_batch.edge_batch
    pyg_batch.subgraph_to_graph = torch.zeros(pyg_batch.num_subgraphs, dtype=torch.long)
    for k, v in pyg_graph:
        if k not in ['x', 'edge_index', 'edge_attr', 'num_nodes', 'batch']:
            pyg_batch[k] = v
    # torch.save(pyg_batch, os.path.join("dataset", batch_path))
    return pyg_batch


def maximal_component(x, edge_attr, edge_index, batch, edge_batch, num_subgraphs):
    # target: only to output the mask but not directly edge index
    largest_node_mask = torch.zeros(1).to(edge_index.device)
    largest_edge_mask = torch.zeros((2, 1)).to(torch.long).to(edge_index.device)
    for i in range(num_subgraphs):
        # get each subgraph
        node_mask = batch == i
        edge_mask = edge_batch == i
        subgraph_x = x[node_mask]
        subgraph_edge_idx = edge_index[:, edge_mask]
        subgraph_edge_attr = edge_attr[edge_mask]
        # get the largest component
        new_subgraph = Data(x=subgraph_x, edge_index=subgraph_edge_idx, edge_attr=subgraph_edge_attr)
        nx_graph = to_networkx(new_subgraph, to_undirected=True)
        del new_subgraph
        largest_component_x = torch.Tensor(list(max(nx.connected_components(nx_graph), key=len))).to(torch.int).to(
            subgraph_edge_idx.device)
        largest_node_mask = torch.cat([largest_node_mask, largest_component_x], dim=0).to(subgraph_edge_idx.device)
        largest_component = torch_geometric.utils.subgraph(largest_component_x, subgraph_edge_idx, subgraph_edge_attr)
        # filter on subgraph and generate a new one
        largest_edge_mask = torch.cat([largest_edge_mask, largest_component[0]], dim=1)

    edge_mask = torch.isin(edge_index[0], largest_edge_mask[0, 1:]) & torch.isin(edge_index[1],
                                                                                 largest_edge_mask[1, 1:])

    return x, edge_index, edge_attr, batch, edge_mask
