import networkx as nx
import numpy as np

G = nx.Graph()
edges = np.array([(1, 3, 1), (2, 3, 2), (3, 4, 0)])
G.add_weighted_edges_from(edges)

matching = nx.max_weight_matching(G)
print(matching)
