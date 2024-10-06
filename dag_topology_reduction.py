import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from contact_plan import IONContactPlanParser
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, TimeExpandedGraph

SHOW_FIGS = True

experiment_name = "mars_earth_s_scenario"

SOURCE_NODES = ["2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012",
                "2013", "2014", "2015", "2016"]
RELAY_NODES = ["3001", "3002", "3003", "3004"]
DESTINATION_NODES = ["1001", "1002"]

EARTH = "EARTH"
MARS = "MARS"

NODE_TO_PLANET_MAP = {
    "1001": EARTH,
    "1002": EARTH,

    "2001": MARS,
    "2002": MARS,
    "2003": MARS,
    "2004": MARS,
    "2005": MARS,
    "2006": MARS,
    "2007": MARS,
    "2008": MARS,
    "2009": MARS,
    "2010": MARS,
    "2011": MARS,
    "2012": MARS,
    "2013": MARS,
    "2014": MARS,
    "2015": MARS,
    "2016": MARS,

    "3001": MARS,
    "3002": MARS,
    "3003": MARS,
    "3004": MARS,
}


def main():
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)

    teg = convert_contact_plan_to_time_expanded_graph(
        contact_plan,
        should_fractionate=True)
    
    reduced_teg = dag_reduction(teg)
    
    print(count_edges(teg), count_edges(reduced_teg))
    print(f"Percent of edges removed = {(1 - count_edges(reduced_teg) / count_edges(teg)):.3f}%")

    visualize(teg)
    visualize(reduced_teg)

    if SHOW_FIGS:
        plt.show()


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
                    tx_node = teg.nodes[tx_idx]
                    rx_node = teg.nodes[rx_idx]

                    # Req. 1
                    is_src_dst = tx_node in SOURCE_NODES and rx_node in DESTINATION_NODES
                    is_src_rly = tx_node in SOURCE_NODES and rx_node in RELAY_NODES
                    is_rly_dst = tx_node in RELAY_NODES and rx_node in DESTINATION_NODES
                    
                    # Req. 2
                    are_nodes_same_planet = NODE_TO_PLANET_MAP[tx_node] == NODE_TO_PLANET_MAP[rx_node]

                    if is_src_dst or (is_src_rly and are_nodes_same_planet) or is_rly_dst:
                        reduced_graph[k][tx_idx][rx_idx] = teg.graphs[k][tx_idx][rx_idx]

    return TimeExpandedGraph(
        graphs=reduced_graph,
        contacts=[],
        state_durations=teg.state_durations,
        K=teg.K,
        N=teg.N,
        nodes=teg.nodes,
        node_map=teg.node_map,
        ipn_node_to_planet_map=teg.ipn_node_to_planet_map,
        W=teg.W)


def visualize(teg):
    rand = np.random.randint(1, 10)
    for k in range(1):
        num_nodes = teg.N

        edges = []
        for tx_idx in range(num_nodes):
            for rx_idx in range(num_nodes):
                if teg.graphs[k][tx_idx][rx_idx] >= 1:
                    edges.append((teg.nodes[tx_idx], teg.nodes[rx_idx]))

        G = nx.DiGraph()
        for node in teg.nodes:
            G.add_node(node)
        G.add_edges_from(edges)

        print(teg.graphs[k])
        plt.figure(k + rand)
        nx.draw(G, nx.spring_layout(G), node_size=1500, with_labels=True)


def count_edges(teg):
    count = 0
    
    for k in range(teg.K):
        for tx_idx in range(teg.N):
            for rx_idx in range(teg.N):
                if teg.graphs[k][tx_idx][rx_idx] >= 1:
                    count += 1
                    
    return count


if __name__ == "__main__":
    main()
