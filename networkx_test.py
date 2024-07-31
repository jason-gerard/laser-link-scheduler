import networkx as nx
import numpy as np

G = nx.Graph()
edges = np.array([(1, 2, 2), (2, 1, 1)])
G.add_weighted_edges_from(edges)

print(nx.adjacency_matrix(G))

matching = nx.max_weight_matching(G)
print(matching)

# a1 = np.array([
#     [1, 2, 3],
#     [1, 2, 3],
#     [1, 2, 3],
# ])
# a2 = np.array([
#     [1, 2, 3],
#     [1, 2, 3],
#     [1, 2, 3],
# ])
# print(3 + a1)
