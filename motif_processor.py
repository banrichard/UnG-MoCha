import math
import os
import pickle
import random

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.graph_operator import load_graph


class QueryPreProcessing(object):
    def __init__(self, queryset_dir: str, true_card_dir: str, dataset: str):
        """
		load the query graphs, true counts, vars
		"""
        self.queryset = queryset_dir
        self.dataset = dataset
        self.data_dir = "dataset"
        self.queryset_load_path = os.path.join(self.data_dir, self.dataset, self.queryset)
        self.true_card_dir = true_card_dir
        self.true_card_load_path = os.path.join(self.data_dir, self.dataset, self.true_card_dir)
        self.num_queries = 0
        self.all_subsets = {}  # {(size, patten) -> [(decomp_graphs, true_card]}
        # preserve the undecomposed queries
        self.all_queries = {}  # {(size, patten) -> [(graph, card)]}
        self.lower_card = 10 ** 0
        self.upper_card = 10 ** 20

    def decomose_queries(self):
        avg_label_den = 0.0
        distinct_card = {}
        subsets_dir = os.listdir(self.queryset_load_path)
        for subset_dir in subsets_dir:
            queries_dir = os.path.join(self.queryset_load_path, subset_dir)
            if not os.path.isdir(queries_dir):
                continue
            pattern, size = subset_dir.split("_")[0], int(subset_dir.split("_")[1])  # Chain(pattern)_size
            self.all_queries[(pattern, size)] = []
            for query_dir in os.listdir(queries_dir):
                query_load_path = os.path.join(self.queryset_load_path, subset_dir, query_dir)
                card_load_path = os.path.join(self.true_card_load_path, subset_dir, query_dir)
                if not os.path.isfile(query_load_path) or os.path.splitext(query_load_path)[1] == ".pickle":
                    continue
                # load, decompose the query
                query, label_den = self.load_query(query_load_path)
                avg_label_den += label_den
                true_card, true_var = self.load_card(card_load_path)
                if true_card >= self.upper_card or true_card < self.lower_card:
                    continue
                true_card = true_card + 1 if true_card == 0 else true_card
                self.all_queries[(pattern, size)].append((query, true_card, true_var))
                self.num_queries += 1
                # save the decomposed query
                query_save_path = os.path.splitext(query_load_path)[0] + ".pickle"
                self.save_decomposed_query(query, true_card, query_save_path)
            # print("save decomposed query: {}".format(query_save_path))
        print("average label density: {}".format(avg_label_den / self.num_queries))

    def node_reorder(self, query, nodes_list, edges_list):
        idx_dict = {}
        node_cnt = 0
        for v in nodes_list:
            idx_dict[v] = node_cnt
            node_cnt += 1
        nodes_list = [(idx_dict[v], {"labels": query.nodes[v]["labels"]})
                      for v in nodes_list]
        edges_list = [(idx_dict[u], idx_dict[v], {"labels": query.edges[u, v]["labels"]})
                      for (u, v) in edges_list]
        sample = nx.Graph()
        sample.add_nodes_from(nodes_list)
        sample.add_edges_from(edges_list)
        return sample

    def load_query(self, query_load_path):
        file = open(query_load_path)
        nodes_list = []
        edges_list = []
        label_cnt = 0
        # query: (nodes, edges,labels)
        for line in file:
            if line.strip().startswith("v"):
                tokens = line.strip().split()
                # v nodeID labelID
                id = int(tokens[1])
                tmp_labels = [int(tokens[2])]  # (only one label in the query node)
                # tmp_labels = [int(token) for token in tokens[2 : ]]
                labels = [] if -1 in tmp_labels else tmp_labels
                label_cnt += len(labels)
                nodes_list.append((id, {"labels": labels}))

            if line.strip().startswith("e"):
                # e srcID dstID labelID1 labelID2....
                tokens = line.strip().split()
                src, dst = int(tokens[1]), int(tokens[2])
                tmp_labels = [float(tokens[3])]
                # tmp_labels = [int(token) for token in tokens[3 : ]]
                labels = [] if -1 in tmp_labels else tmp_labels
                edges_list.append((src, dst, {"labels": labels}))

        query = nx.Graph()
        query.add_nodes_from(nodes_list)
        query.add_edges_from(edges_list)

        # print('number of nodes: {}'.format(graph.number_of_nodes()))
        # print('number of edges: {}'.format(graph.number_of_edges()))
        file.close()
        label_den = float(label_cnt) / query.number_of_nodes()
        return query, label_den

    def load_card(self, card_load_path):
        with open(card_load_path, "r") as in_file:
            token = in_file.readline().split(" ")
            # card = in_file.readline().strip()
            card, var = float(token[0]), float(token[1])
            in_file.close()
        return card, var

    def save_decomposed_query(self, graphs, card, save_path):
        with open(save_path, "wb") as out_file:
            obj = {"graphs": graphs, "card": card}
            pickle.dump(obj=obj, file=out_file, protocol=3)
            out_file.close()


class Queryset(object):
    def __init__(self, dataset_name, data_dir, dataset, all_queries):
        """
        all_queries: {(size, patten) -> [(graphs, true_card, ture_var]} // all queries subset
        """

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.dataset = dataset
        self.num_queries = 0
        self.graph = load_graph(
            os.path.join(self.data_dir, self.dataset,
                         self.dataset_name))  # "../dataset/krogan/krogan_core.txt"
        self.node_label_card, self.edge_label_card = self.get_label_card(self.graph)
        self.node_label_fre = 0
        self.edge_label_fre = 0
        self.label_dict = {key: key for key in self.node_label_card.keys()}
        # embed_feat_path = os.path.join(args.embed_feat_dir, "{}.csv".format(args.dataset_name))
        # embed_feat = np.loadtxt(embed_feat_path, delimiter=" ")[:, 1:]
        #
        # # assert embed_feat.shape[0] == len(self.node_label_card) + 1, "prone embedding size error!"
        # self.embed_dim = embed_feat.shape[1]
        # self.embed_feat = torch.from_numpy(embed_feat)

        self.num_node_feat = len(self.node_label_card)

        self.edge_embed_feat = None
        self.edge_embed_dim = 0
        self.num_edge_feat = len(self.edge_label_card)
        self.all_subsets = self.transform_motif_to_tensors(all_queries)
        self.all_sizes = {}
        for (pattern, size), graphs_card_pairs in self.all_subsets.items():
            if size not in self.all_sizes.keys():
                self.all_sizes[size] = []
            self.all_sizes[size] += graphs_card_pairs

        self.num_train_queries, self.num_val_queries, self.num_test_queries = 0, 0, 0
        train_sets, val_sets, test_sets, all_train_sets = self.data_split(self.all_sizes, train_ratio=0.8,
                                                                          val_ratio=0.1)
        self.train_loaders = self.to_dataloader(all_sets=train_sets)
        self.val_loaders = self.to_dataloader(all_sets=val_sets)
        self.test_loaders = self.to_dataloader(all_sets=test_sets)
        self.all_train_loaders = self.to_dataloader(all_sets=all_train_sets)
        self.train_sets, self.val_sets, self.test_sets, self.all_train_sets = train_sets, val_sets, test_sets, all_train_sets

    def transform_motif_to_tensors(self, all_queries):
        tmp_subsets = {}
        for (pattern, size), graphs_card_pairs in all_queries.items():
            tmp_subsets[(pattern, size)] = []
            for (motif, card, var) in graphs_card_pairs:
                decomp_x, decomp_edge_index, decomp_edge_weight = \
                    self._get_motif_data(motif)
                tmp_subsets[(pattern, size)].append((decomp_x, decomp_edge_index, decomp_edge_weight, card, var))
                self.num_queries += 1

        return tmp_subsets

    def _get_motif_data(self, motif):
        x = []
        edge_index = None
        edge_attr = None
        node_attr = self._get_nodes_attr_freq(motif)
        edge_index, edge_attr = self._get_edge_weight_freq(motif)
        x = node_attr
        return x, edge_index, edge_attr

    def _get_nodes_attr_freq(self, motif):
        node_attr = torch.ones(size=(motif.number_of_nodes(), len(self.node_label_card)), dtype=torch.float)
        for v in motif.nodes():
            for label in motif.nodes[v]["labels"]:
                node_attr[v][self.label_dict[label]] = self.node_label_card[label]
                self.node_label_fre += 1
        return node_attr

    # def _get_nodes_attr_embed(self, motif):
    #     node_attr = torch.zeros(size=(motif.number_of_nodes(), self.embed_dim), dtype=torch.float)
    #     for v in motif.nodes():
    #         if len(motif.nodes[v]["labels"]) == 0:
    #             continue
    #         for label in motif.nodes[v]["labels"]:
    #             node_attr[v] += self.embed_feat[self.label_dict[label]]
    #             self.node_label_fre += 1
    #     return node_attr

    def _get_edges_index_freq(self, motif):
        edge_index = torch.ones(size=(2, motif.number_of_edges()), dtype=torch.long)
        edge_attr = torch.zeros(size=(motif.number_of_edges(), len(self.edge_label_card)), dtype=torch.float)
        cnt = 0
        for e in motif.edges():
            edge_index[0][cnt], edge_index[1][cnt] = e[0], e[1]

            for label in motif.edges[e]["labels"]:
                edge_attr[cnt] = self.edge_label_card[label]
                self.edge_label_fre += 1
            cnt += 1
        return edge_index, edge_attr

    def _get_edge_weight_freq(self, motif):
        edge_index = torch.ones(size=(2, motif.number_of_edges()), dtype=torch.long)
        edge_weight = torch.zeros(size=(motif.number_of_edges(), 1), dtype=torch.float)
        cnt = 0
        for (u, v, wt) in motif.edges.data('labels'):
            edge_index[0][cnt], edge_index[1][cnt] = u, v
            edge_weight[cnt] = wt[-1]
            cnt += 1
        return edge_index, edge_weight

    def get_label_card(self, graph: nx.Graph):
        node_label_card = {}
        edge_label_card = {}
        for node in graph.nodes(data=True):
            if len(node[-1]) == 0:
                node_label_card[-1] = node_label_card.get(-1, 0) + 1.0
            else:
                for fea in node[-1]:
                    node_label_card[fea] += 1.0
        for edge in graph.edges(data=True):
            if len(edge[-1]) == 0:
                edge_label_card[-1] += edge_label_card.get(-1, 0) + 1.0
            else:
                edge_label_card[edge[-1]['prob']] = edge_label_card.get(edge[-1]['prob'], 0) + 1.0
        for key, val in node_label_card.items():
            node_label_card[key] = val / graph.number_of_nodes()
        for key, val in edge_label_card.items():
            edge_label_card[key] = val / graph.number_of_edges()
        return node_label_card, edge_label_card

    def data_split(self, all_sets, train_ratio, val_ratio, seed=1):
        assert train_ratio + val_ratio <= 1.0, "Error data split ratio!"
        random.seed(seed)
        train_sets, val_sets, test_sets = [], [], []
        all_train_sets = [[]]
        for key in sorted(all_sets.keys()):
            num_instances = len(all_sets[key])
            random.shuffle(all_sets[key])
            train_sets.append(all_sets[key][: int(num_instances * train_ratio)])
            # merge to all_train_sets
            all_train_sets[-1] = all_train_sets[-1] + train_sets[-1]
            val_sets.append(
                all_sets[key][int(num_instances * train_ratio): int(num_instances * (train_ratio + val_ratio))])
            test_sets.append(all_sets[key][int(num_instances * (train_ratio + val_ratio)):])
            self.num_train_queries += len(train_sets[-1])
            self.num_val_queries += len(val_sets[-1])
            self.num_test_queries += len(test_sets[-1])
        return train_sets, val_sets, test_sets, all_train_sets

    def to_dataloader(self, all_sets, batch_size=1, shuffle=True):
        datasets = [QueryDataset(queries=queries)
                    for queries in all_sets]
        dataloaders = [DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
                       for dataset in datasets]
        return dataloaders


class QueryDataset(Dataset):
    def __init__(self, queries):
        """
        parameter:
        query =[(x, edge_index, edge_attr, card, var)]
        """
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        """
        decomp_x, decomp_edge_attr, decomp_edge_attr: list[Tensor]
        """
        x, edge_index, edge_attr, card, var = self.queries[index]
        card = torch.tensor(math.log2(card), dtype=torch.float)
        var = torch.tensor(math.log2(var), dtype=torch.float)

        return x, edge_index, edge_attr, card, var


def _to_datasets(all_sets):
    datasets = [QueryDataset(queries=queries)
                for queries in all_sets] if isinstance(all_sets, list) \
        else [QueryDataset(queries=all_sets)]
    return datasets


def _to_dataloaders(datasets, batch_size=1, shuffle=True):
    """
    create a lists of torch dataloader from datasets
    """
    dataloaders = [DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
                   for dataset in datasets] if isinstance(datasets, list) \
        else [DataLoader(dataset=datasets, batch_size=batch_size, shuffle=shuffle)]
    return dataloaders
