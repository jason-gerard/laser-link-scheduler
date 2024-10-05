import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from contact_plan import IONContactPlanParser
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, TimeExpandedGraph

SHOW_FIGS = False

experiment_name = "mars_earth_xs_scenario"

SOURCE_NODES = ["2001", "2002", "2003", "2004"]
RELAY_NODES = ["3001"]
DESTINATION_NODES = ["1001"]

contact_plan_parser = IONContactPlanParser()
contact_plan = contact_plan_parser.read(experiment_name)

teg = convert_contact_plan_to_time_expanded_graph(
    contact_plan,
    should_fractionate=True)


def dag_reduction(teg: TimeExpandedGraph):
    """
    The directed-acyclic graph (DAG) topology reduction algorithm follows several rules and cases to remove cycles
    and reduce the number of edges in the graph
    Requirement 1: The edge is a part of one of the follow path types: S -> D (one hop) and S -> R -> D (two hops), then
                   the specific edge types to be kept include: S -> D, S -> R, R -> D
    Requirement 2: The source and relay nodes are both orbiting the same planet for any S -> R edge
    """
    reduced_graph = np.zeros((teg.K, teg.N, teg.N), dtype="int64")
    
    for k in range(teg.K):
        for tx_idx in range(teg.N):
            for rx_idx in range(teg.N):
                if teg.graphs[k][tx_idx][rx_idx] >= 1:
                    # Req. 1
                    is_src_dst = True
                    is_src_rly = True
                    is_rly_dst = True
                    
                    # Req. 2
                    is_rly_same_planet = True

                    if is_src_dst or (is_src_rly and is_rly_same_planet) or is_rly_dst:
                        reduced_graph[k][tx_idx][rx_idx] = teg.graphs[k][tx_idx][rx_idx]


def visualize():
    for k in range(10):
        num_nodes = teg.N

        edges = []
        for tx_idx in range(num_nodes):
            for rx_idx in range(num_nodes):
                if teg.graphs[k][tx_idx][rx_idx] >= 1:
                    edges.append((teg.nodes[tx_idx], teg.nodes[rx_idx]))

        G = nx.Graph()
        for node in teg.nodes:
            G.add_node(node)
        G.add_edges_from(edges)

        # print(teg.graphs[k])
        plt.figure(k)
        nx.draw(G, with_labels=True)

    if SHOW_FIGS:
        plt.show()
