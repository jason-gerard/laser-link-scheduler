import networkx as nx
import argparse
import copy
import pprint
from timeit import default_timer as timer
from tqdm import tqdm

import numpy as np

import constants
from weights import delta_time, compute_node_capacities, compute_capacity, compute_wasted_capacity, \
    compute_node_capacity_by_graph, delta_capacity, merge_many_node_capacities
from contact_plan import IONContactPlanParser, Contact
from time_expanded_graph import build_time_expanded_graph, write_time_expanded_graph, \
    TimeExpandedGraph, convert_time_expanded_graph_to_contact_plan, Graph, time_expanded_graph_splitter
from utils import FileType


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--experiment_name', help='Name of experiment folder')
    return parser.parse_args()


def main(experiment_name):
    start = timer()

    # Read contact plan from disk
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)
    print("Finished reading contact plan")

    # Convert contact plan into a time expanded graph (TEG)
    time_expanded_graph = build_time_expanded_graph(contact_plan)
    write_time_expanded_graph(experiment_name, time_expanded_graph, FileType.TEG)
    print("Finished converting contact plan to time expanded graph")

    # Split long contacts in the TEG into multiple smaller contacts, this will result in each k state having a maximum
    # duration of d_max
    split_time_expanded_graph = time_expanded_graph_splitter(time_expanded_graph)
    write_time_expanded_graph(experiment_name, split_time_expanded_graph, FileType.SPLIT)
    print("Finished splitting time expanded graph")

    # Iterate through each of the k graphs and compute the maximal matching
    # As we step through the k graphs we want to optimize for our metric
    # This will produce the TEG with the maximum Earth-bound network capacity
    print("Starting scheduling contacts")
    scheduled_time_expanded_graph = lls(split_time_expanded_graph)
    write_time_expanded_graph(experiment_name, scheduled_time_expanded_graph, FileType.TEG_SCHEDULED)
    print("Finished scheduling contacts")
    
    node_capacities = compute_node_capacities(scheduled_time_expanded_graph)
    network_capacity = compute_capacity(node_capacities)
    print(f"Scheduled network capacity: {network_capacity}")
    network_wasted_capacity = compute_wasted_capacity(node_capacities)
    print(f"Scheduled network wasted capacity: {network_wasted_capacity}")

    # Convert the TEG back to a contact plan
    scheduled_contact_plan = convert_time_expanded_graph_to_contact_plan(scheduled_time_expanded_graph)
    contact_plan_parser.write(experiment_name, scheduled_contact_plan, FileType.SCHEDULED)
    print("Finished converting time expanded graph to contact plan")
    
    end = timer()
    print(f"Elapsed time: {end - start} seconds")


def lls(time_expanded_graph: TimeExpandedGraph) -> TimeExpandedGraph:
    """
    This algorithm is a max-weight maximal matching

    The weights of the matrix W_k should be calculated such that the weight of each edge should correspond to the
    delta capacity or delta wasted capacity if that edge was selected plus the sum of time that contact was
    previously disabled for

    delta_cap -> The increased network capacity (max weight maximal matching) or wasted capacity (min weight maximal
    matching) if the edge i,j is selected for step k.
    
    Inputs: contact topology [P] of size K x N x N
            IPN node mappings [X] of size N
            state durations [T] of size K
    Outputs: contact plan [L] of size K x N x N
    
    for k <- 0 to K do
      [W]_k,i,j <- delta_capacity([P]_k, [L], [X])
                   + alpha * delta_time([P]_k, [L]_k, [T]) for all i,j
      Blossom([P]_k, [L]_k, [W]_k)
    """

    scheduled_graphs = []
    
    num_nodes = len(time_expanded_graph.nodes)
    node_capacities = []
    W_delta_time = np.zeros((num_nodes, num_nodes), dtype=int)

    for graph in tqdm(time_expanded_graph.graphs):
        # Compute weights based on the current contact topology, P_k, the contact plans for the previous processed
        # states, L, and the list of interplanetary nodes, X.
        # TODO convert this to a weight sum i.e. (1-a)*c + a*t
        # With the previous states node capacities + the possible combinations for this current state, compute the new
        # capacities
        delta_cap = delta_capacity(graph.adj_matrix, node_capacities, time_expanded_graph.ipn_node_to_planet_map, graph.state_duration)
        W_k = delta_cap + (constants.alpha * W_delta_time)

        # Create list of edges, represented by three-tuple of (tx_idx, rx_idx, weight) based on the contact topology P_k
        # and computed weights based on delta_capacity + alpha * delta_time
        edges = []
        for tx_idx in range(num_nodes):
            for rx_idx in range(num_nodes):
                if graph.adj_matrix[tx_idx][rx_idx] >= 1:
                    edges.append((tx_idx, rx_idx, W_k[tx_idx][rx_idx]))
        
        # Create graph containing edges from P_k
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        matched_edges = nx.max_weight_matching(G)

        # Compute L_k from the matched edges
        # Build adj_matrix from matched edges list. nx.max_weight_matching works on an undirected graph so when we see
        # an edge add it in both directions i.e. (i,j) and (j,i)
        adj_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for tx_idx, rx_idx in matched_edges:
            # Make sure to map the value of the adj_matrix, the communication interface back correctly
            adj_matrix[tx_idx][rx_idx] = graph.adj_matrix[tx_idx][rx_idx]
            adj_matrix[rx_idx][tx_idx] = graph.adj_matrix[rx_idx][tx_idx]
            
        filtered_contacts = [contact for contact in graph.contacts if should_keep_contact(time_expanded_graph, matched_edges, contact)]

        scheduled_graphs.append(Graph(
            contacts=filtered_contacts,
            adj_matrix=adj_matrix,
            k=graph.k,
            state_duration=graph.state_duration,
            state_start_time=graph.state_start_time,))
        
        # Update node_capacities list with node capacities from state k contact plan and merge them together
        selected_node_capacities = compute_node_capacity_by_graph(
            np.array(adj_matrix),
            graph.state_duration,
            time_expanded_graph.ipn_node_to_planet_map)
        node_capacities = merge_many_node_capacities(node_capacities + selected_node_capacities)
        
        # Update the matrix containing the disabled contact time for state k
        W_delta_time += delta_time(graph.adj_matrix, adj_matrix, graph.state_duration)

    return TimeExpandedGraph(
        graphs=scheduled_graphs,
        nodes=time_expanded_graph.nodes,
        interplanetary_nodes=time_expanded_graph.interplanetary_nodes,
        ipn_node_to_planet_map=time_expanded_graph.ipn_node_to_planet_map,
        node_map=time_expanded_graph.node_map,
        start_time=time_expanded_graph.start_time,
        end_time=time_expanded_graph.end_time,
    )


def should_keep_contact(time_expanded_graph: TimeExpandedGraph, matched_edges: set, contact: Contact) -> bool:
    tx_idx = time_expanded_graph.node_map[contact.tx_node]
    rx_idx = time_expanded_graph.node_map[contact.rx_node]
    
    return (tx_idx, rx_idx) in matched_edges or (rx_idx, tx_idx) in matched_edges


def fair_contact_plan(time_expanded_graph: TimeExpandedGraph) -> TimeExpandedGraph:
    """
    Max-weight maximal matching
    Inputs: contact topology [P] of size K x N x N
            state durations [T] of size K
    Outputs: contact plan [L] of size K x N x N

    DCT_i,j <- 0 for all i,j
    for k <- 0 to K do
      [W]_k,i,j <- DCT_i,j for all i,j
      Blossom([P]_k, [L]_k, [W]_k)
      if [L]_k,i,j = 0 then
        DCT_i,j <- DCT_i,j + [T]_k for all i,j
    """
    pass


if __name__ == "__main__":
    args = get_args()
    main(args.experiment_name)
