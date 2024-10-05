import pulp

bit_rate = 1

max_matching = 2
nodes = "A B C D".split()
source_nodes = "A B".split()
destination_nodes = ["D"]
relay_nodes = ["C"]

# T = [100, 100, 100, 100]
T = [100, 200, 100]
K = len(T)

possible_edges = [tuple([idx_t, c[0], c[1]]) for c in pulp.allcombinations(nodes, max_matching) if len(c) == 2 for idx_t in range(K)]
print(possible_edges)


edges = pulp.LpVariable.dicts(
    "edges", possible_edges, lowBound=0, upBound=1, cat=pulp.LpInteger
)

flow_model = pulp.LpProblem("Network_flow_model", pulp.LpMaximize)


def flow(i, j):
    selected_edges = [edge for edge in possible_edges if i in edge and j in edge]
    return sum([edges[edge] * T[edge[0]] * bit_rate for edge in selected_edges])


capacity = pulp.LpVariable("Capacity", lowBound=0)

# objective function
flow_model += capacity

# inflow
flow_model += capacity <= pulp.lpSum([flow(i, relay_node) for i in source_nodes for relay_node in relay_nodes])
# outflow
flow_model += capacity <= pulp.lpSum([flow(relay_node, x) for x in destination_nodes for relay_node in relay_nodes])

# constraint that each node can only be a part of a single selected edge per state
for node in nodes:
    for k in range(K):
        flow_model += (
            pulp.lpSum([edges[edge] for edge in possible_edges if node in edge and k in edge]) <= 1,
            f"Max_edge_{node}_{k}",
        )

# TODO add constraint that a selected edge must of part of the initial contact plan, this might just be better done by
# filtering the list of possible edges, same as I plan to do for the DAG topology reduction

flow_model.solve()

for edge in possible_edges:
    if edges[edge].value() == 1.0:
        print(edge)
