import pulp

bit_rate = 1

max_matching = 2
# nodes = "A B C D".split()
# source_nodes = "A B".split()
# destination_nodes = ["D"]
# relay_nodes = ["C"]
nodes = "A B C D E".split()
source_nodes = "A B".split()
destination_nodes = ["D"]
relay_nodes = ["C", "E"]

# T = [100, 100, 100, 100]
T = [100, 200, 100]
# T = [100, 100, 100]
K = len(T)


def flow(edges, i, j):
    selected_edges = [edge for edge in edges.keys() if i in edge and j in edge]
    return sum([edges[edge] * T[edge[0]] * bit_rate for edge in selected_edges])


def dct(edges, i):
    """
    In order to make the schedule fair to all the nodes we can use the disabled contact time (DCT) for each inflow edge
    by taking the total duration of the schedule minus the enabled contact time
    """
    schedule_duration = sum(T)
    selected_edges = [edge for edge in edges.keys() if i in edge]
    # T[edge[0]] gives the duration of the edge
    return sum([schedule_duration - edges[edge] * T[edge[0]] for edge in selected_edges])


def solver():
    possible_edges = [tuple([idx_t, c[0], c[1]]) for c in pulp.allcombinations(nodes, max_matching) if len(c) == max_matching for
                      idx_t in range(K)]
    print(possible_edges)

    # This represents the constraint that a selected edge must be a part of the initial contact plan
    edges = pulp.LpVariable.dicts(
        "edges", possible_edges, lowBound=0, upBound=1, cat=pulp.LpInteger
    )

    flow_model = pulp.LpProblem("Network_flow_model", pulp.LpMaximize)

    capacities = {relay_node: pulp.LpVariable(f"Capacity_{relay_node}", lowBound=0) for relay_node in relay_nodes}

    # objective function should maximize the summation of the capacity for each relay satellite
    flow_model += pulp.lpSum(capacities.values())

    for relay_node, capacity in capacities.items():
        # inflow
        flow_model += capacity <= pulp.lpSum([flow(edges, i, relay_node) for i in source_nodes])
        # outflow
        flow_model += capacity <= pulp.lpSum([flow(edges, relay_node, x) for x in destination_nodes])

    # constraint that each node can only be a part of a single selected edge per state
    for node in nodes:
        for k in range(K):
            flow_model += (
                pulp.lpSum([edges[edge] for edge in edges.keys() if node in edge and k in edge]) <= 1,
                f"Max_edge_{node}_{k}",
            )

    flow_model.solve()

    for edge in edges.keys():
        if edges[edge].value() == 1.0:
            print(edge)


if __name__ == "__main__":
    solver()
