import random

import numpy as np
import numpy.random
import pandas as pd
import sys
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import queue
import os
from networkx.algorithms import chordal


def load_data(file):
    # raw_data_np = np.loadtxt(file, skiprows=1)
    # print(raw_data_np)
    # num_node,num_edge = raw_data.iloc[0]
    # nodes = raw_data_np[:, :-1]
    # features = raw_data_np[:, -1]
    g = nx.Graph()

    with open(file, 'r') as f:
        next(f)  # skip the first line of dataset
        for line in f:
            token = line.rstrip().split(" ")
            src, des, prob = int(token[0]), int(token[1]), float(token[-1])
            g.add_edge(src, des, prob=prob)
    return g


def split_data(data: str, strategy: str):
    """
    Randomly select the node index to split the dataset into training,validation and testing set according to the strategy

    :param data:
    :param strategy:
    :return:
    """
    data_array = np.genfromtxt("dataset/simulation/" + data + ".txt")
    data_size = np.max(data_array) + 1
    training_size = int(int(strategy[0:2]) / 100 * data_size)
    validation_size = int(int(strategy[2:4]) / 100 * data_size)
    test_size = int(data_size - (training_size + validation_size))
    id_train = []
    id_val = []
    id_test = []
    id_list = [x for x in range(int(data_size))]
    while len(id_train) < training_size:
        id = id_list.pop(random.randint(0, len(id_list) - 1))
        id_train.append(id)
    while len(id_val) < validation_size:
        id = id_list.pop(random.randint(0, len(id_list) - 1))
        id_val.append(id)
    while len(id_test) < test_size:
        id = id_list.pop(random.randint(0, len(id_list) - 1))
        id_test.append(id)
    return id_train, id_val, id_test


def prepare_data_pack(graph: nx.Graph, strategies: list, data: str, data_dir: str):
    """
    Create the split dataset and save as dataset_strategy.pkl file
    :param graph: the graph to load
    :param strategies: temporarily 602020 or 051580 according to EGNN paper
    :param data: the dataset to be split
    :param data_dir: the dataset dir
    """
    node_id = list(graph.nodes)
    # read the nodes matrix X
    X = np.array(node_id)
    # todo: get Y from LINC
    Y = np.array(nx.get_edge_attributes(graph, 'prob'))
    A = nx.adjacency_matrix(graph, node_id)

    with open(data_dir + "/" + data + ".pkl", 'wb') as f:
        pickle.dump((X, Y, A), f)
    node_id2idx = dict(zip(node_id, range(
        len(node_id))))  # zip to a new dictionary {node: 'id'} to extract selected nodes and dataset splitting
    for splitting in strategies:
        id_train, id_val, id_test = split_data(data, splitting)
        idx_train = [node_id2idx[i] for i in id_train]
        idx_val = [node_id2idx[i] for i in id_val]
        idx_test = [node_id2idx[i] for i in id_test]
        with open(data_dir + "/" + (data + '-tvt-' + splitting) + ".pkl", 'wb') as f:
            pickle.dump((idx_train, idx_val, idx_test), f)


def save_graph_pickle(file, graph):
    pickle.dump(graph, open(file, 'wb'))


def graph_generate(file_dir, node_num: int, m) -> nx.Graph:
    """
    node_num: number of nodes

    :return: a random graph with node_num vertices and edge_num edges
    """
    G = nx.barabasi_albert_graph(node_num, m)
    for i in range(len(list(G.nodes))):
        G.nodes[i]['label'] = -1
    adj = G.adj
    e = [
        (u, v, {"prob": random.gauss(0.68, 0.06)})
        for u, nbrs in adj.items()
        for v, d in nbrs.items()
    ]
    G.update(edges=e, nodes=adj)
    save_graph_pickle(file_dir + "s1_test.pkl", G)

    with open("dataset/simulation/s1_t_1.txt", 'w') as f:
        f.write("t # 0\n")
        for node in G.nodes(data=True):
            f.write("v {} {}\n".format(node[0], node[1]['label']))
        for edge in G.edges(data=True):
            f.write("e {} {} {:.2f}\n".format(edge[0], edge[1], edge[2]['prob']))
        print("data graph is generated!")
    return G


def to_LSS_format(graph, file):
    for i in range(len(list(graph.nodes))):
        graph.nodes[i]['label'] = -1
    with open("dataset/intel/" + file, 'w') as f:
        f.write("t # 0\n")
        for node in graph.nodes(data=True):
            f.write("v {} {}\n".format(node[0], node[1]['label']))
        for edge in graph.edges(data=True):
            f.write("e {} {} {:.2f}\n".format(edge[0], edge[1], edge[2]['prob']))
        print("data graph is generated!")
    return graph


class QuerySampler(object):
    def __init__(self, graph):
        self.graph = graph

    def sample_star(self, node_num):
        nodes_list = []
        edges_list = []
        while True:
            src = random.randint(0, self.graph.number_of_nodes())
            if self.graph.degree[src] >= node_num - 1:
                break
        nodes_list.append(src)
        nexts = random.sample(list(self.graph.neighbors(src)), k=node_num - 1)
        for v in nexts:
            nodes_list.append(v)
            edges_list.append((src, v))
        sample = self.node_reorder(nodes_list, edges_list)
        return sample

    def sample_tree(self, node_num):
        nodes_list = []
        edges_list = []
        parent = {}

        src = random.randint(0, self.graph.number_of_nodes())
        Q = queue.Queue()
        Q.put(src)
        while not Q.empty():
            cur = Q.get()
            if len(nodes_list) > 0:
                edges_list.append((parent[cur], cur))
            nodes_list.append(cur)
            if len(nodes_list) == node_num:
                break

            candidates = set(list(self.graph.neighbors(cur))).difference(set(nodes_list))
            if len(candidates) == 0:
                continue
            nexts = random.sample(list(self.graph.neighbors(src)),
                                  k=random.randint(1, min(len(candidates), node_num - len(nodes_list))))
            for v in nexts:
                Q.put(v)
                parent[v] = cur

        sample = self.node_reorder(nodes_list, edges_list)
        return sample

    def sample_triangle(self, node_num):
        nodes_list = []
        edges_list = []
        while True:
            src = random.randint(0, self.graph.number_of_nodes())
            neighbors = list(graph.neighbors(src))
            if len(neighbors) >= 2:
                for v in neighbors:
                    v_neigbor = graph.neighbors(v)
                    v_set = set(v_neigbor)
                    common_neighbors = v_set.intersection(set(neighbors))
                    if len(common_neighbors) == 0:
                        continue
                    else:
                        w = random.choice(list(common_neighbors))
                        nodes_list.append(src)
                        nodes_list.append(v)
                        nodes_list.append(w)
                        edges_list.append((src, v))
                        edges_list.append((v, w))
                        edges_list.append((src, w))
                        if node_num > 3:
                            tail_node = nodes_list[random.randint(0, len(nodes_list) - 1)]
                            neighbor_candidates = set(graph.neighbors(tail_node)).difference(set(nodes_list))
                            candidate = list(neighbor_candidates)[random.randint(0, len(neighbor_candidates) - 1)]
                            nodes_list.append(candidate)
                            edges_list.append((tail_node, candidate))
                        sample = self.node_reorder(nodes_list, edges_list)
                        return sample

    def sample_cycle(self, node_num):
        circles = nx.cycle_basis(self.graph)
        selected_circles = [circle for circle in circles if len(circle) == node_num]
        nodes_list = list(selected_circles[random.randint(0, len(selected_circles))])
        subgraph = nx.subgraph(self.graph, nodes_list)
        edges_list = list(subgraph.edges())
        sample = self.node_reorder(nodes_list, edges_list)
        return sample

    def sample_chordal_cycle(self, node_num):

        cliques = list(nx.chordal_graph_cliques(self.graph))
        candidates = [clique for clique in cliques if len(clique) == node_num]
        nodes_list = candidates[random.randint(0, len(candidates) - 1)]
        subgraph = nx.subgraph(self.graph, nodes_list)
        edges_list = []

        sample = sampler.node_reorder(nodes_list, edges_list)
        return sample

    def sample_path(self, node_num):
        nodes_list = []
        edges_list = []
        parent = {}
        q = queue.Queue()
        src = random.randint(0, self.graph.number_of_nodes())
        q.put(src)
        while not q.empty():
            if len(nodes_list) == node_num:
                break
            cur = q.get()
            if len(nodes_list) > 0:
                edges_list.append((parent[cur], cur))
            nodes_list.append(cur)
            candidates = set(list(self.graph.neighbors(cur))).difference(set(nodes_list))
            if len(candidates) == 0:
                continue
            next = random.sample(list(candidates), k=1)[0]

            q.put(next)
            parent[next] = cur
        sample = self.node_reorder(nodes_list, edges_list)
        return sample

    def sample_clique(self, node_num):
        cliques = list(nx.find_cliques(self.graph))
        candidates = [clique for clique in cliques if len(clique) == node_num]
        nodes_list = candidates[random.randint(0, len(candidates) - 1)]
        edge_list = nx.subgraph(self.graph, nodes_list).edges()
        sample = self.node_reorder(nodes_list, edge_list)
        # nodes_list = []
        # edges_list = []
        # for v in range(0, node_num):
        #     nodes_list.append((v, {"label": -1, "dvid": -1}))
        #     for u in range(0, v):
        #         edges_list.append((u, v, {"prob": self.graph.edges[u, v]["prob"]}))
        # sample = nx.Graph()
        # sample.add_nodes_from(nodes_list)
        # sample.add_edges_from(edges_list)
        return sample

    def node_reorder(self, nodes_list, edges_list):
        idx_dict = {}
        node_cnt = 0
        for v in nodes_list:
            idx_dict[v] = node_cnt
            node_cnt += 1
        nodes_list = [(idx_dict[v], {"label": -1, "dvid": -1})
                      for v in nodes_list]
        edges_list = [(idx_dict[u], idx_dict[v], {"prob": self.graph.edges[u, v]["prob"]})
                      for (u, v) in edges_list]
        sample = nx.Graph()
        sample.add_nodes_from(nodes_list)
        sample.add_edges_from(edges_list)
        return sample


# id_train, id_val, id_test = split_data("collins.txt", "602020")
# print(id_train)
# print(id_val)
# print(id_test)
# g = graph_generate("./dataset/simulation/", 200, 2)
# with open("dataset/simulation/s1_test.pkl", 'rb') as f:
#     graph = pickle.load(f)
#     # edge_list = nx.to_pandas_edgelist(graph)
#     # edge_list.to_csv("dataset/simulation/s14linc.txt", sep=" ", header=None, index=False,float_format="%.2f")
graph = load_data("dataset/intel/intel.txt")
graph = to_LSS_format(graph, "intel_lss.txt")
sampler = QuerySampler(graph)
label_dict = {'star_3': [189.2791, 902.7179],
              'triangle_3': [1483.7243, 18453.2766],
              'path_4': [49.2315, 393.9129],
              'star_4': [504.5776, 15707.4893],
              'tailedtriangle_4': [2961.2061, 317758.7999],
              'cycle_4': [3613.9587, 561663.3236],
              'clique_4': [367.7054, 5148.8938]}
for i in range(100):
    sample = sampler.sample_star(3)
    with open("dataset/intel/queryset/star_3/{}.txt".format(i), 'w') as f:
        f.write("t # {}\n".format(i))
        for node in sample.nodes(data=True):
            f.write("v {} {} {}\n".format(node[0], node[1]["label"], node[1]['dvid']))
        for edge in sample.edges(data=True):
            f.write("e {} {} {:.2f}\n".format(edge[0], edge[1], edge[2]['prob']))
    with open("dataset/intel/label/star_3/{}.txt".format(i), 'w') as f:
        f.write(str(label_dict['star_3'][0]) + " " + str(label_dict['star_3'][1]))
# prepare_data_pack(g, ["602020"], "s1", "dataset/simulation")
