import os.path as osp
import torch
import pandas as pd

from torch_geometric.data import InMemoryDataset, Data


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

        x = pd.read_csv("dataset/krogan/embedding/krogan_embeded_sorted.csv", header=None, delimiter=" ").iloc[:,1:].values
        x = torch.from_numpy(x)
        edge_index = torch.from_numpy(edge_index)
        edge_attr = torch.from_numpy(edge_attr)
        num_nodes = len(x)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        data, slices = self.collate([graph])
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    graph = Converter()
    data = graph[0]
    print(data.num_nodes)
    # batch = create_subgraphs(data)
    # tmp = Batch.from_data_list(list(data))
    # new_data = create_subgraphs(data)
    # print(new_data.num_features)
    # new_data = create_subgraphs(data)
    # subgraphs,mask_tensor = subgraph_padding(graph[0].sets[0])
    # print(subgraphs)
    # sets = k_hop_random_walk(graph)
    # new_datas= nodes_to_subgraphs(graph[0])
    # topk = SimpleTopK()
    # tmp = topk.top_k_scores(data,3)
    # print(tmp)
    # x = pd.read_csv("dataset/krogan/embedding/krogan_embeded.csv", header=None, skiprows=1, delimiter=" ")
    # x = x.sort_values(by=x.columns[0])
    # x.to_csv("dataset/krogan/embedding/krogan_embeded_sorted.csv", header=False, index=False, sep=" ")
