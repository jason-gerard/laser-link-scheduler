import networkx as nx
import numpy as np
from tqdm import tqdm

import constants
from contact_plan import Contact
from time_expanded_graph import TimeExpandedGraph, Graph
from weights import delta_time, compute_node_capacity_by_graph, delta_capacity, merge_many_node_capacities


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
        # Compute the change in network capacity on an edge by edge basis using the previous states node capacities and
        # the possible choices or decisions of active edges for this current state
        W_delta_cap = delta_capacity(
            graph.adj_matrix,
            node_capacities,
            time_expanded_graph.ipn_node_to_planet_map,
            graph.state_duration)
        
        # Compute the weight of each edge by doing a weighted sum of the capacity and fairness metrics
        W_k = ((1 - constants.alpha) * W_delta_cap) + (constants.alpha * W_delta_time)

        # Compute max weight maximal matching using the blossom algorithm
        matched_edges = blossom(graph.adj_matrix, W_k)

        # Compute L_k from the matched edges
        L_k = build_graph(matched_edges, graph, time_expanded_graph)
        scheduled_graphs.append(L_k)
        
        # Update node_capacities list with node capacities from state k contact plan and merge them together
        scheduled_node_capacities = compute_node_capacity_by_graph(
            L_k.adj_matrix,
            graph.state_duration,
            time_expanded_graph.ipn_node_to_planet_map)
        node_capacities = merge_many_node_capacities(node_capacities + scheduled_node_capacities)
        
        # Update the matrix containing the disabled contact time for state k
        W_delta_time += delta_time(graph.adj_matrix, L_k.adj_matrix, graph.state_duration)

    return TimeExpandedGraph(
        graphs=scheduled_graphs,
        nodes=time_expanded_graph.nodes,
        ipn_node_to_planet_map=time_expanded_graph.ipn_node_to_planet_map,
        node_map=time_expanded_graph.node_map,
        start_time=time_expanded_graph.start_time,
        end_time=time_expanded_graph.end_time,
    )


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


def blossom(P_k: np.ndarray, W_k: np.ndarray) -> set:
    num_nodes = len(P_k)
    
    # Create list of edges, represented by three-tuple of (tx_idx, rx_idx, weight) based on the contact topology P_k
    # and computed weights based on delta_capacity + alpha * delta_time
    edges = []
    for tx_idx in range(num_nodes):
        for rx_idx in range(num_nodes):
            if P_k[tx_idx][rx_idx] >= 1:
                # When we compute the weight matrix it is not symmetric because we compute the capacity on an edge basis
                # but since the networkx lib uses a symmetric matrix in order to not write the 0 edge weight to the
                # graph on accident, because the weight of the last edge added will be used, we can just take the max of
                # the edge weights
                max_weight = max(W_k[tx_idx][rx_idx], W_k[rx_idx][tx_idx])
                edges.append((tx_idx, rx_idx, max_weight))

    # Create graph containing edges from P_k
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    
    # Perform max weight matching using the blossom algorithm. We leverage the networkx library to do this
    return nx.max_weight_matching(G)


def build_graph(matched_edges: set, contact_topology_k: Graph, time_expanded_graph: TimeExpandedGraph) -> Graph:
    num_nodes = len(contact_topology_k.adj_matrix)
    # Build adj_matrix from matched edges list. nx.max_weight_matching works on an undirected graph so when we see
    # an edge add it in both directions i.e. (i,j) and (j,i)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for tx_idx, rx_idx in matched_edges:
        # Make sure to map the value of the adj_matrix, the communication interface back correctly
        adj_matrix[tx_idx][rx_idx] = contact_topology_k.adj_matrix[tx_idx][rx_idx]
        adj_matrix[rx_idx][tx_idx] = contact_topology_k.adj_matrix[rx_idx][tx_idx]

    filtered_contacts = [contact for contact in contact_topology_k.contacts if
                         should_keep_contact(time_expanded_graph, matched_edges, contact)]

    return Graph(
        contacts=filtered_contacts,
        adj_matrix=adj_matrix,
        k=contact_topology_k.k,
        state_duration=contact_topology_k.state_duration,
        state_start_time=contact_topology_k.state_start_time)


def should_keep_contact(time_expanded_graph: TimeExpandedGraph, matched_edges: set, contact: Contact) -> bool:
    tx_idx = time_expanded_graph.node_map[contact.tx_node]
    rx_idx = time_expanded_graph.node_map[contact.rx_node]

    return (tx_idx, rx_idx) in matched_edges or (rx_idx, tx_idx) in matched_edges
