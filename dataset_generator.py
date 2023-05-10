from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from itertools import repeat
from tqdm import tqdm


class PygGraphPropPredDataset(InMemoryDataset):
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
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.filepath)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name)

    @property
    def raw_file_names(self):
        return ['./dataset/krogan/label', "./dataset/krogan/queryset"]

    @property
    def processed_file_names(self):
        return ['data.pt']

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
        datalist = []
        labels = self.raw_file_names[0]
        querysets = self.raw_file_names[1]

        def iterate_folders(folder1_path, folder2_path):
            for (root1, dirs1, files1), (root2, dirs2, files2) in zip(os.walk(folder1_path), os.walk(folder2_path)):
                # Iterate over files in both folders simultaneously
                for file1, file2 in zip(files1, files2):
                    queryset = pd.read_csv(osp.join(root1, file1), header=None, skiprows=4, delimiter=" ")
                    edge_index = torch.from_numpy(queryset.iloc[:, 1:3].values.reshape(2, -1))
                    edge_attr = torch.from_numpy(queryset.iloc[:, -1].values.reshape(-1, 1))

                    label = pd.read_csv(osp.join(root2, file2), header=None, delimiter=" ")
                    mean = torch.from_numpy(label[0].values.reshape(-1, 1))
                    var = torch.from_numpy(label[1].values.reshape(-1, 1))

                    g = Data(edge_index=edge_index, edge_attr=edge_attr, mean=mean, var=var)
                    datalist.append(g)
                # Recurse into subfolders
                for subfolder1, subfolder2 in zip(dirs1, dirs2):
                    iterate_folders(os.path.join(root1, subfolder1), os.path.join(root2, subfolder2))

        iterate_folders(querysets, labels)

        data, slices = self.collate(datalist)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    dataset = PygGraphPropPredDataset()
