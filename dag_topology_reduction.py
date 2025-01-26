import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from constants import SOURCE_NODES, DESTINATION_NODES, RELAY_NODES, NODE_TO_PLANET_MAP, EARTH
from contact_plan import IONContactPlanParser
from time_expanded_graph import convert_contact_plan_to_time_expanded_graph, TimeExpandedGraph

SHOW_FIGS = True

EXPERIMENT_NAME = "mars_earth_xs_scenario"


def main():
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(EXPERIMENT_NAME)

    teg = convert_contact_plan_to_time_expanded_graph(
        contact_plan,
        should_fractionate=True)
    
    reduced_teg = dag_reduction(teg)
    
    print(count_edges(teg), count_edges(reduced_teg))
    print(f"Percent of edges removed = {100 * (1 - count_edges(reduced_teg) / count_edges(teg)):.3f}%")

    if SHOW_FIGS:
        visualize(teg, name="teg")
        visualize(reduced_teg, name="reduced_teg")
        # plt.show()


def dag_reduction(teg: TimeExpandedGraph):
    """
    The directed-acyclic graph (DAG) topology reduction algorithm follows several rules and cases to remove cycles
    and reduce the number of edges in the graph
    Requirement 1: The edge is a part of one of the follow path types: S -> D (one hop) and S -> R -> D (two hops), then
                   the specific edge types to be kept include: S -> D, S -> R, R -> D
    Requirement 2: The source and relay nodes are both orbiting the same planet for any S -> R edge or if the relay node
                   is orbiting the destination planet
    """
    reduced_graph = np.zeros((teg.K, teg.N, teg.N), dtype="int64")

    print("Starting the DAG topology reduction algorithm")
    for k in tqdm(range(teg.K)):
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
                    is_rly_on_dst_planet = NODE_TO_PLANET_MAP[rx_node] == EARTH

                    if is_src_dst or (is_src_rly and (are_nodes_same_planet or is_rly_on_dst_planet)) or is_rly_dst:
                        reduced_graph[k][tx_idx][rx_idx] = teg.graphs[k][tx_idx][rx_idx]

    reduced_teg = TimeExpandedGraph(
        graphs=reduced_graph,
        contacts=teg.contacts,
        state_durations=teg.state_durations,
        K=teg.K,
        N=teg.N,
        nodes=teg.nodes,
        node_map=teg.node_map,
        ipn_node_to_planet_map=teg.ipn_node_to_planet_map,
        W=teg.W,
        pos=teg.pos)

    print(count_edges(teg), count_edges(reduced_teg))
    print(f"Percent of edges removed = {100 * (1 - count_edges(reduced_teg) / count_edges(teg)):.3f}%")
    
    return reduced_teg


def visualize(teg, name):
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

        # print(teg.graphs[k])
        plt.figure(k + rand)
        # nx.draw(G, nx.spring_layout(G), node_size=1500, with_labels=False)
        A = nx.nx_agraph.to_agraph(G)
        A.layout(prog="dot")
        A.draw(f'{name}.png', args='-Gnodesep=0.01 -Gfont_size=1', prog='dot')

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
