from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from itertools import repeat
import torch_geometric.transforms as T
from torch_geometric.data.data import BaseData
from tqdm import tqdm
from torch_geometric.utils import from_networkx

from utils.graph_operator import load_graph, k_hop_induced_subgraph_edge


class UGDataset(InMemoryDataset):
    def __init__(self, root="./dataset", name="krogan_core", transform=None, pre_transform=None, pre_filter=None,
                 use_edge_attr=True, filepath="krogan", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        self.root = root
        self.use_edge_attr = use_edge_attr
        self.filepath = filepath
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.cnt = 0
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.filepath)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.filepath, self.name)

    @property
    def raw_file_names(self):
        return ['./dataset/krogan/label', "./dataset/krogan/queryset"]

    @property
    def processed_file_names(self):
        return [self.name + '.pt']

    def dataset_split(self):
        assert self.train_ratio + self.val_ratio <= 1.0, "Error split ratios!"
        dataset.shuffle()
        train_idx = []
        valid_idx = []
        test_idx = []
        train_size = int(dataset.len() * self.train_ratio)
        val_size = int(dataset.len() * self.val_ratio)
        test_size = int(dataset.len() * self.test_ratio)
        for i in range(train_size):
            train_idx.append(i)
        for i in range(train_size, train_size + val_size):
            valid_idx.append(i)
        for i in range(train_size + val_size, dataset.len()):
            test_idx.append(i)
        return {'train': torch.tensor(train_idx, dtype=torch.long), 'valid': torch.tensor(valid_idx, dtype=torch.long),
                'test': torch.tensor(test_idx, dtype=torch.long)}

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        graph = load_graph(osp.join(self.root, self.filepath, self.name + ".txt"))
        edge_count = {}
        for edge in graph.edges(data=True):
            subgraph = k_hop_induced_subgraph_edge(graph, edge)
            pyg_subgraph = from_networkx(subgraph, group_edge_attrs=all)
            if pyg_subgraph.num_nodes == 0:
                continue
            pyg_subgraph.edge_attr = pyg_subgraph.edge_attr.expand(-1, 2).clone()
            data_list.append(pyg_subgraph)
            for i in range(pyg_subgraph.edge_index.size(1)):
                edge = pyg_subgraph.edge_index[:, i]
                edge_count[tuple(edge.numpy())] = edge_count.get(tuple(edge.numpy()), 0) + 1
            # need to update after computing the edge count
            self.cnt += 1
        for subgraph in data_list:
            for j in range(subgraph.edge_index.size(1)):
                edge = tuple(subgraph.edge_index[:, j].numpy())
                subgraph.edge_attr[j] = torch.cat([subgraph.edge_attr[j, 1].view(-1, 1),
                                                      torch.tensor(edge_count.get(edge)).view(-1, 1).to(
                                                          subgraph.edge_attr.dtype)], dim=1)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = UGDataset()
    trans = T.LargestConnectedComponents()
    dataloader = DataLoader(dataset, batch_size=dataset.len(), shuffle=False)
    for batch in dataloader:
        batch = trans(batch)
