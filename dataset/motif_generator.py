from torch_geometric.datasets.motif_generator.custom import CustomMotif
import networkx as nx

nodes = [0, 1, 2, 3, 4, 5]
edges = [(0, 1, {"prob": 0.1}), (0, 2, {"prob": 0.1}), (0, 3, {"prob": 0.1}), (0, 4, {"prob": 0.1}),
         (0, 5, {"prob": 0.1}), (1, 2, {"prob": 0.1}), (1, 3, {"prob": 0.1}), (1, 4, {"prob": 0.1}),
         (1, 5, {"prob": 0.1}), (2, 3, {"prob": 0.1}), (2, 4, {"prob": 0.1}), (2, 5, {"prob": 0.1}),
         (3, 4, {"prob": 0.1}),
         (3, 5, {"prob": 0.1}), (4, 5, {"prob": 0.1})]
graph = nx.Graph()
graph.add_nodes_from(nodes)
graph.add_edges_from(edges)
custom = CustomMotif(graph)
print(custom.structure)
