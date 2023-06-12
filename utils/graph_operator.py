import queue
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric.utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from utils.batch import Batch
import torch.nn.functional as F


# from dataloader import DataLoader


def load_graph(filepath) -> nx.Graph:
    nx_graph = nx.Graph()
    # print(nx_graph.number_of_edges())
    edges = []
    graph_data = pd.read_csv(filepath, header=None, skiprows=1, delimiter=" ")
    for i in range(len(graph_data)):
        edges.append((graph_data.iloc[i, 0], graph_data.iloc[i, 1], {'prob': graph_data.iloc[i, 2]}))
    nx_graph.add_edges_from(edges)
    return nx_graph


def k_hop_induced_subgraph_edge(graph, edge, k=1):
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
    edge_list = [(u, v, {"prob": graph.edges[u, v]["prob"]})
                 for (u, v) in edge_list]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(node_list)
    subgraph.add_edges_from(edge_list)
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
    subgraph = nx.subgraph(graph, nodes_list).copy()

    return subgraph


def candidate_filter(candidate_set):
    final_candidate = None
    prob = 0
    for candidate in candidate_set:
        cur_prob = 1
        for e in candidate.edges(data=True):
            cur_prob *= e[2]['prob']
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
            if tmp_graph.edges[u, v]['prob'] < random.random():
                remove_edge_list.append((u, v))
        tmp_graph.remove_edges_from(remove_edge_list)
        remove_node_list = [node for node in tmp_graph.nodes() if tmp_graph.degree(node) == 0]
        tmp_graph.remove_nodes_from(remove_node_list)
        candidate_set.append(tmp_graph)
    subgraph_with_highest_probability = candidate_filter(candidate_set)
    return subgraph_with_highest_probability


def create_batch(graph: nx.Graph, candidate_sets: dict, emb_path=None):
    """
    For current stage, only support 1-hop subgraph(s)
    :param graph: the original graph (the node features are pre-embedded by Node2Vec)
    :param candidate_sets: the candidate subgraph(s) of each node
    :return: a Batch contains subgraph representation
    """
    # convert networkx graph into pyg style
    # edge_attrs = list(graph.edges.data("prob"))
    # pyg_graph = from_networkx(graph,group_edge_attrs=['prob'])
    if emb_path is not None:
        x = torch.from_numpy(np.loadtxt(emb_path, delimiter=" ")[:, 1:])
    else:
        x = torch.zeros(len(graph.nodes), 1)
        degree = list(graph.degree)
        for i in range(len(degree)):
            x[i] = degree[i][1]

        # x = torch.ones([graph.number_of_nodes(), 1])
        # x = nx.to_numpy_array(graph,weight='prob')
        # x = torch.from_numpy(x)

    pyg_graph = torch_geometric.utils.from_networkx(graph)
    pyg_graph.x = x
    # get the corresponding subgraph(s) of each node in the graph
    # for now we randomly select a subgraph (2023.5.15 21:30)
    pyg_subgraphs = []
    for node in range(graph.number_of_nodes()):
        subgraph = candidate_sets[node]
        pyg_subgraph = torch_geometric.utils.from_networkx(subgraph)
        pyg_subgraphs.append(pyg_subgraph)
    pyg_batch = Batch.from_data_list(pyg_subgraphs)

    pyg_batch.edge_index = torch.LongTensor(pyg_batch.edge_index)
    pyg_batch.num_nodes = sum(data_.num_nodes for data_ in pyg_subgraphs)
    pyg_batch.num_subgraphs = len(pyg_subgraphs)

    pyg_batch.original_x = pyg_graph.x
    pyg_batch.original_edge_index = pyg_graph.edge_index
    pyg_batch.original_edge_attr = pyg_graph.edge_attr
    pyg_batch.node_to_subgraph = pyg_batch.batch
    del pyg_batch.batch
    pyg_batch.subgraph_to_graph = torch.zeros(len(pyg_subgraphs), dtype=torch.long)
    return pyg_batch


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    graph = load_graph("../dataset/krogan/krogan_core.txt")
    # subgraph = k_hop_induced_subgraph(graph, 0)
    # candidate_sets = generate_candidate_sets(subgraph, 0)
    candidate_sets = {}
    for edge in graph.edges(data=True):
        subgraph = k_hop_induced_subgraph_edge(graph, edge)
        print(subgraph.edges())
        nx.draw(subgraph, with_labels=True)
        plt.show()
        break
    for node in range(graph.number_of_nodes()):
        subgraph = k_hop_induced_subgraph(graph, node)
        nx.draw(subgraph, with_labels=True)
        plt.show()
        break
    #     candidate_sets[node] = random_walk_on_subgraph(subgraph, node)
    # start_time = time.time()
    # batch = create_batch(graph, candidate_sets)
    # end_time = time.time()
    # # torch.save(batch,"../dataset/krogan/graph_batch.pt")
    # print("running time of batch creation is {}s".format(end_time - start_time))
    #
    # # print(candidate_sets[0].edges)
    # # print(len(candidate_sets[0].edges))
    # batch = torch.load("../model/graph_batch.pt")
    # print(batch)
    # #
    # loader = DataLoader(batch, batch_size=1, shuffle=False)
    # #
    # batch = next(iter(loader))
