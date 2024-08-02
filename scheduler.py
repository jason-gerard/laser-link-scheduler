import networkx as nx
import numpy as np
from tqdm import tqdm

import constants
from contact_plan import Contact
from time_expanded_graph import TimeExpandedGraph
from weights import delta_time, compute_node_capacity_by_graph, delta_capacity, merge_many_node_capacities


class LaserLinkScheduler:
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
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
        constants.metrics["W_avg"] = np.zeros((teg.N, teg.N))  # Used for reporting only

        scheduled_graphs = np.zeros((teg.K, teg.N, teg.N), dtype='int64')
        scheduled_contacts = []

        node_capacities = []
        W_delta_time = np.zeros((teg.N, teg.N), dtype='int64')

        for k in tqdm(range(teg.K)):
            # Compute the change in network capacity on an edge by edge basis using the previous states node capacities and
            # the possible choices or decisions of active edges for this current state
            W_delta_cap = delta_capacity(
                teg.graphs[k],
                node_capacities,
                teg.ipn_node_to_planet_map,
                teg.state_durations[k])

            # Compute the weight of each edge by doing a weighted sum of the capacity and fairness metrics
            W_k = ((1 - constants.alpha) * W_delta_cap) + (constants.alpha * W_delta_time)

            constants.metrics["W_avg"] += W_k  # Used for reporting only

            # Compute max weight maximal matching using the blossom algorithm
            matched_edges = blossom(teg.graphs[k], W_k)

            # Compute L_k from the matched edges
            L_k, contacts = build_graph(matched_edges, teg.graphs[k], teg.contacts[k], teg.node_map)
            scheduled_graphs[k] = L_k
            scheduled_contacts.append(contacts)

            # Update node_capacities list with node capacities from state k contact plan and merge them together
            scheduled_node_capacities = compute_node_capacity_by_graph(
                L_k,
                teg.state_durations[k],
                teg.ipn_node_to_planet_map)
            node_capacities = merge_many_node_capacities(node_capacities + scheduled_node_capacities)

            # Update the matrix containing the disabled contact time for state k
            W_delta_time += delta_time(teg.graphs[k], L_k, teg.state_durations[k])

        constants.metrics["W_avg"] = (constants.metrics["W_avg"] / teg.K).tolist()  # Used for reporting only

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            ipn_node_to_planet_map=teg.ipn_node_to_planet_map)


class FairContactPlan:
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
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
        return teg


class RandomScheduler:
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        """
        Apply blossom algorithm with no weights
        """
        return teg


def blossom(P_k: np.ndarray, W_k: np.ndarray) -> set:
    """
    The blossom algorithm assumes undirected edges meaning that we cannot have A -> B without B -> A. Logically this
    makes sense for laser communications because both laser transceivers must be physically pointing towards each other.
    This algorithm does support individually defining the properties of the laser in each direction i.e. A -> B with
    laser ID 1 but B -> A with laser ID 3.
    """
    num_nodes = len(P_k)

    # Create list of edges, represented by three-tuple of (tx_idx, rx_idx, weight) based on the contact topology P_k
    # and computed weights based on delta_capacity + alpha * delta_time
    edges = []
    for tx_idx in range(num_nodes):
        for rx_idx in range(num_nodes):
            if P_k[tx_idx][rx_idx] >= 1:
                # When we compute the weight matrix it is not symmetric because we compute the capacity on an edge basis
                # but since the networkx lib uses a symmetric matrix it will select both edges. To account for this we
                # sum the weights of the edges in either direction to become the total weight for that undirected edge.
                total_weight = W_k[tx_idx][rx_idx] + W_k[rx_idx][tx_idx]
                edges.append((tx_idx, rx_idx, total_weight))

    # Create graph containing edges from P_k
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # Perform max weight matching using the blossom algorithm. We leverage the networkx library to do this
    return nx.max_weight_matching(G)


def build_graph(
        matched_edges: set,
        contact_topology_k: np.ndarray,
        contacts_k: list[Contact],
        node_map: dict[str, int]
) -> tuple[np.ndarray, list[Contact]]:
    num_nodes = len(contact_topology_k)
    # Build adj_matrix from matched edges list. nx.max_weight_matching works on an undirected graph so when we see
    # an edge add it in both directions i.e. (i,j) and (j,i)
    contact_plan_k = np.zeros((num_nodes, num_nodes), dtype='int64')
    for tx_idx, rx_idx in matched_edges:
        # Make sure to map the value of the graph i.e. the communication interface id back to the correct edge. This
        # allows us to support different lasers in each direction while using an undirected graph algorithm (blossom)
        contact_plan_k[tx_idx][rx_idx] = contact_topology_k[tx_idx][rx_idx]
        contact_plan_k[rx_idx][tx_idx] = contact_topology_k[rx_idx][tx_idx]

    contacts = [contact for contact in contacts_k if should_keep_contact(matched_edges, node_map, contact)]

    return contact_plan_k, contacts


def should_keep_contact(matched_edges: set, node_map: dict[str, int], contact: Contact) -> bool:
    node1_idx = node_map[contact.tx_node]
    node2_idx = node_map[contact.rx_node]

    return (node1_idx, node2_idx) in matched_edges or (node2_idx, node1_idx) in matched_edges
