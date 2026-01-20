import math
import networkx as nx
import numpy as np
from itertools import combinations, chain, product

from laser_link_scheduler.topology.contact_plan import IONContactPlanParser
from laser_link_scheduler.graph.time_expanded_graph import (
    convert_contact_plan_to_time_expanded_graph,
)


def all_maximal_matchings(T):
    maximal_matchings = []
    partial_matchings = [{(u, v)} for (u, v) in T.edges()]

    while partial_matchings:
        # get current partial matching
        m = partial_matchings.pop()
        nodes_m = set(chain(*m))

        extended = False
        for u, v in T.edges():
            if u not in nodes_m and v not in nodes_m:
                extended = True
                # copy m, extend it and add it to the list of partial matchings
                m_extended = set(m)
                m_extended.add((u, v))
                partial_matchings.append(m_extended)

        if not extended and m not in maximal_matchings:
            maximal_matchings.append(m)

    return maximal_matchings


EXPERIMENT_NAME = "gs_mars_earth_xs_scenario"

contact_plan_parser = IONContactPlanParser()
contact_plan = contact_plan_parser.read(EXPERIMENT_NAME)

time_expanded_graph = convert_contact_plan_to_time_expanded_graph(
    contact_plan, should_fractionate=True
)

possible_graphs = []
# Tested with max k at 7 states, anything past that takes too long to run
for k in range(time_expanded_graph.K):
    # for k in range(5):
    edges = []
    for tx_idx in range(time_expanded_graph.N):
        for rx_idx in range(time_expanded_graph.N):
            if time_expanded_graph.graphs[k][tx_idx][rx_idx] >= 1:
                edges.append((tx_idx, rx_idx))

    G = nx.Graph()
    G.add_edges_from(edges)

    possible_graphs.append(all_maximal_matchings(G))

print("Valid maximal matchings per state", [len(g) for g in possible_graphs])
print(sum([len(g) for g in possible_graphs]))
print("Number of possible solutions:", math.prod([len(g) for g in possible_graphs]))
solutions = product(*possible_graphs)

# Next we would need to go and compute the capacity of each possible solution and take the best one.
