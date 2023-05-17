import queue
import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

from batch import Batch
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

    return subgraph.copy()


def random_walk_on_subgraph(subgraph: nx.Graph, node, walks=10, subs=1):
    # Set random seed
    random.seed(6324)
    candidate_set = []
    # Generate 1-hop subgraphs using random walk
    for s in range(subs):
        node_list = [node]
        for i in range(walks):
            neighbors = list(graph.neighbors(node))
            if len(neighbors) == 0:
                break
            next_node = neighbors[random.randint(0, len(neighbors) - 1)]
            node_list.append(next_node)
        node_list = set(node_list)
        tmp_graph = subgraph.subgraph(list(node_list)).copy()
        remove_node_list = []
        remove_edge_list = []
        for (u, v) in tmp_graph.edges():
            if tmp_graph.edges[u, v]['prob'] < random.random():
                remove_edge_list.append((u, v))
        tmp_graph.remove_edges_from(remove_edge_list)
        remove_node_list = [node for node in tmp_graph.nodes() if tmp_graph.degree(node) == 0]
        tmp_graph.remove_nodes_from(remove_node_list)
        candidate_set.append(tmp_graph)
    return candidate_set


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
        x = torch.ones([graph.number_of_nodes(), 1])
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    edge_attr = torch.tensor([dict(graph[u][v])['prob'] for u, v in graph.edges]).view(-1, 1).contiguous()
    pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # get the corresponding subgraph(s) of each node in the graph
    # for now we randomly select a subgraph (2023.5.15 21:30)
    pyg_subgraphs = []
    for node in range(graph.number_of_nodes()):
        subgraph = candidate_sets[node]
        # if subgraph.number_of_edges() != 0:
        sub_edge_index = torch.tensor(list(subgraph.edges)).t().contiguous()
        sub_edge_attr = torch.tensor([dict(subgraph[u][v])['prob'] for u, v in subgraph.edges]).view(-1,
                                                                                                     1).contiguous()
        sub_x = x[list(subgraph.nodes()), :]
        pyg_subgraph = Data(x=sub_x, edge_index=sub_edge_index, edge_attr=sub_edge_attr)
            # pyg_subgraph = from_networkx(subgraph, group_edge_attrs=['prob'])

        # for n in subgraph.nodes():
        #     subgraph.add_node(n,idx = n)
        # else:
        #     # pyg_subgraph = Data(x = torch.tensor())
        #     pyg_subgraph = from_networkx(subgraph)
        # add attributes to the subgraph

        pyg_subgraphs.append(pyg_subgraph)
    pyg_batch = Batch.from_data_list(pyg_subgraphs)
    pyg_batch.num_nodes = sum(data_.num_nodes for data_ in pyg_subgraphs)
    pyg_batch.num_subgraphs = len(pyg_subgraphs)

    pyg_batch.original_edge_index = edge_index
    pyg_batch.original_edge_attr = edge_attr
    pyg_batch.node_to_subgraph = pyg_batch.batch
    pyg_batch.subgraph_to_graph = torch.zeros(len(pyg_batch), dtype=torch.long)
    return pyg_batch


# graph = load_graph("../dataset/krogan/krogan_core.txt")
# # subgraph = k_hop_induced_subgraph(graph, 0)
# # candidate_sets = generate_candidate_sets(subgraph, 0)
# candidate_sets = {}
# for node in range(graph.number_of_nodes()):
#     subgraph = k_hop_induced_subgraph(graph, node)
#     candidate_set = random_walk_on_subgraph(subgraph, node)
#     candidate_sets[node] = candidate_set[random.randint(0, len(candidate_set) - 1)]
# start_time = time.time()
# batch = create_batch(graph, candidate_sets, "../dataset/krogan/embedding/krogan_embeded_sorted.csv")
# end_time = time.time()
# torch.save(batch,"../dataset/krogan/graph_batch.pt")
# print("running time of batch creation is {}s".format(end_time - start_time))
# print(batch)

# print(candidate_sets[0].edges)
# print(len(candidate_sets[0].edges))
batch = torch.load("../dataset/krogan/graph_batch.pt")

loader = DataLoader(batch,batch_size=1,shuffle=False)
#
batch = next(iter(loader))