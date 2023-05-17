import os
import pickle

import networkx as nx


class QueryDecompose(object):
    def __init__(self, queryset_dir: str, true_card_dir: str, dataset: str):
        """
		load the query graphs, true counts, vars
		"""
        self.queryset = queryset_dir
        self.dataset = dataset
        self.queryset_load_path = os.path.join(dataset, queryset_dir)
        self.true_card_dir = true_card_dir
        self.true_card_load_path = os.path.join(dataset, true_card_dir)
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
    def __init__(self, args, all_queries):
        """
        all_queries: {(size, patten) -> [(graphs, true_card, ture_var]} // all queries subset
        """
        self.args = args
        self.data_dir = args.data_dir
        self.dataset = args.dataset
        self.num_queries = 0
        self.all_subsets = self.transform_motif_to_tensors(all_queries)

    def transform_motif_to_tensors(self, all_queries):
        tmp_subsets = {}
        for (pattern, size), graphs_card_pairs in all_queries.items():
            tmp_subsets[(pattern, size)] = []
            for (motifs, card, var) in graphs_card_pairs:
                decomp_x, decomp_edge_index, decomp_edge_weight, _ = \
                    self._get_motif_data(motifs)
                tmp_subsets[(pattern, size)].append((decomp_x, decomp_edge_index, decomp_edge_weight, card, var))
                self.num_queries += 1

        return tmp_subsets

    def _get_motif_data(self, motifs):
        x = []
        edge_index = []
        edge_attr = []
        for motif in motifs:
            if self.args.embed_type == "freq":
                node_attr = self._get_nodes_attr_freq(motif)
            elif self.args.embed_type == "n2v" or self.args.embed_type == "prone": # node2vec
                node_attr = self._get_nodes_attr_embed(motif)
            if self.args.edge_embed_type == "freq":
                edge_index, edge_attr = self._get_edges_index_freq(motif)
                if self.args.model_type == "GCN2" or self.args.model_type == "FA" or self.args.model_type == "GCN" or self.args.model_type == "Graph":
                    edge_index, edge_attr = self._get_edge_weight_freq(motif)
    def _get_nodes_attr_freq(self, motif):
        pass

    def _get_nodes_attr_embed(self, motif):
        pass

    def _get_edges_index_freq(self, motif):
        pass

    def _get_edge_weight_freq(self, motif):
        pass