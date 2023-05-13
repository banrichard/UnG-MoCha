import os.path as osp
import torch
import pandas as pd

from torch_geometric.data import InMemoryDataset, Data

from model.topK import SimpleTopK
from utils.batch import Batch
from utils.utils import k_hop_random_walk, nodes_to_subgraphs, subgraph_padding, create_subgraphs


class Converter(InMemoryDataset):
    """
    read the whole graph and convert into subgraphs

    subgraphs -> candidate sets of subgraphs on each node
    """

    def __init__(self, root="./dataset", name="krogan_graph", transform=None, pre_transform=None, pre_filter=None,
                 use_edge_attr=True, filepath="krogan", filename="korgan_core.txt"):
        self.root = root
        self.use_edge_attr = use_edge_attr
        self.filepath = filepath
        self.name = name
        self.filename = filename
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.filepath)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def processed_file_names(self):
        return ['whole_graph.pt']

    def process(self):
        graph_data = pd.read_csv("dataset/krogan/krogan_core.txt", header=None, skiprows=1, delimiter=" ")
        edge_index = graph_data.iloc[:, 0:2].values.T
        edge_attr = graph_data.iloc[:, -1].values.reshape(-1, 1)
        num_nodes = len(set(graph_data.iloc[:, 0]).union(set(graph_data.iloc[:, 1])))
        edge_index = torch.from_numpy(edge_index)
        edge_attr = torch.from_numpy(edge_attr)
        graph = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        new_data = create_subgraphs(graph)
        data, slices = self.collate([new_data])
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    graph = Converter()
    data = graph[0]
    # tmp = Batch.from_data_list(list(data))
    # new_data = create_subgraphs(data)
    # print(new_data.num_features)
    # new_data = create_subgraphs(data)
    # subgraphs,mask_tensor = subgraph_padding(graph[0].sets[0])
    # print(subgraphs)
    # sets = k_hop_random_walk(graph)
    # new_datas= nodes_to_subgraphs(graph[0])
    # topk = SimpleTopK(graph[0].sets)
    # tmp = topk.top_k_subgraphs(topk.candidate_set)
    # print(tmp)