import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.data.separate import separate
from torch_geometric.data.batch import Batch

# This is a copy from torch_geometric/data/batch.py
# which is modified to support batch assignment in subgraph level

class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]  # get x, edge_idx, edge_attr
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        batch.edge_batch = []
        # sizes = [len(subgraph.x) for subgraph in data_list]
        # data_list = [subgraph for _, subgraph in sorted(zip(sizes, data_list), reverse=True)]
        # sizes.sort(reverse=True)
        # read from each subgraph

        for i, data in enumerate(data_list):
            for key in data.keys:  # x,edge_idx,edge_attr
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(key, data[key]))  # item.size(0) = 17
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])

                if key == 'node_to_subgraph':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                elif key == 'subgraph_to_graph':
                    cumsum[key] = cumsum[key] + 1
                elif key == 'original_edge_index':
                    cumsum[key] = cumsum[key] + data.num_subgraphs
                else:
                    cumsum[key] = cumsum[key] + data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size,), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            num_edges = data.num_edges
            if num_nodes is not None:
                item = torch.full((num_nodes,), i, dtype=torch.long)
                batch.batch.append(item)
            if num_edges is not None:
                edge_item = torch.full((num_edges,), i, dtype=torch.long)
                batch.edge_batch.append(edge_item)
        if num_nodes is None:
            batch.batch = None
        if num_edges is None:
            batch.edge_batch = None
        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])

        if torch_geometric.is_debug_enabled():
            batch.debug()
        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using Batch.from_data_list()'))

        keys = ['x', 'edge_index', 'edge_attr']
        cumsum = {key: 0 for key in keys}
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            data = self.__data_class__()
            for key in keys:
                if torch.is_tensor(self[key]):
                    try:
                        data[key] = self[key].narrow(
                            data.__cat_dim__(key,
                                             self[key]), self.__slices__[key][i],
                            self.__slices__[key][i + 1] - self.__slices__[key][i])
                    except RuntimeError:
                        continue
                    if self[key].dtype != torch.bool:
                        data[key] = data[key] - cumsum[key]
                else:
                    data[key] = self[key][self.__slices__[key][i]:self.__slices__[key][i + 1]]
                cumsum[key] = cumsum[key] + data.__inc__(key, data[key])
            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


if __name__ == "__main__":
    batch = []
    x2 = torch.tensor([[0], [1], [2]], dtype=torch.float)
    edge_index2 = torch.tensor([[0, 1, 2], [1, 0, 1]], dtype=torch.long)
    edge_attr2 = torch.tensor([[0.2], [0.3], [0.4]], dtype=torch.float)
    batch.append(Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2))
    x3 = torch.tensor([[3], [4], [5], [6]], dtype=torch.float)
    edge_index3 = torch.tensor([[3, 4, 5, 6], [4, 5, 6, 3]], dtype=torch.long)
    edge_attr3 = torch.tensor([[0.2], [0.3], [0.4], [0.5]], dtype=torch.float)
    batch.append(Data(x=x3, edge_index=edge_index3, edge_attr=edge_attr3))
    pyg_batch = Batch.from_data_list(batch)
    pyg_batches = pyg_batch.to_data_list()
    print(pyg_batches)
    # topk_sample = TopKEdgePooling(ratio=0.5)
    # y = topk_sample(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
