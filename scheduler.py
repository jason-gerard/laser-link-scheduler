import networkx as nx
from networkx.algorithms.bipartite.matching import minimum_weight_full_matching
import numpy as np
from tqdm import tqdm

import constants
from contact_plan import Contact
from time_expanded_graph import TimeExpandedGraph
from weights import disabled_contact_time, compute_node_capacity_by_graph, delta_capacity, merge_many_node_capacities


class LaserLinkScheduler:
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        """
        This algorithm is a max-weight maximal matching, where it will iterate through each of the k graphs and
        compute the maximal matching. By maximizing the capacity model at each of the k graphs we will produce the
        TEG with the maximum Earth-bound network capacity. The weights of the matrix W_k should be calculated such
        that the weight of each edge should correspond to the delta capacity or delta wasted capacity if that edge
        was selected plus the sum of time that contact was disabled.

        Inputs: contact topology [P] of size K x N x N
                IPN node mappings [X] of size N
                state durations [T] of size K
        Outputs: contact plan [L] of size K x N x N
        
        for k <- 0 to K do
          d_c <- delta_capacity([P]_k, [L], [X])
          [W]_k,i,j <- (1 - a) * d_c + a * dct
          Blossom([P]_k, [L]_k, [W]_k)
          dct <- delta_time([P]_k, [L]_k, [T])
        """
        scheduled_graphs = np.empty((teg.K, teg.N, teg.N), dtype='int64')
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        node_capacities = []
        W_dct = np.zeros((teg.N, teg.N), dtype='int64')

        for k in tqdm(range(teg.K)):
            # Compute the change in network capacity on an edge by edge basis using the previous states node
            # capacities and the possible choices or decisions of active edges for this current state. This is a
            # dynamic programming approach where we used the memoized values of the weights of the previous k states
            # to compute the new weights matrix for capacity for state k+1.
            # We pass in the previous graphs as previous state, from there we can see for the two nodes making the edge
            # where were they looking before and where will they look now.
            # In the case that a node did not have a link in the previous time slice we can assume they are still
            # pointing at the last node they were in contact with.
            W_delta_cap = delta_capacity(
                teg.graphs[k],
                scheduled_graphs[:k],
                node_capacities,
                teg.nodes,
                teg.state_durations[k],
                teg.pos,
                teg.optical_interfaces_to_node,
                teg.node_to_optical_interfaces
            )

            # Compute the weight of each edge by doing a weighted sum of the capacity and fairness metrics
            weights[k] = ((1 - constants.alpha) * W_delta_cap) + (constants.alpha * W_dct)

            # Compute max weight maximal matching using the blossom algorithm
            matched_edges = blossom(teg.graphs[k], weights[k])

            # Compute L_k from the matched edges
            L_k, contacts = build_graph(matched_edges, teg.graphs[k], teg.contacts[k], teg.node_map)
            scheduled_graphs[k] = L_k
            scheduled_contacts.append(contacts)

            # Update node_capacities list with node capacities from state k contact plan and merge them together
            scheduled_node_capacities = compute_node_capacity_by_graph(
                L_k,
                teg.state_durations[k],
                teg.nodes,
                scheduled_graphs[:k],
                teg.pos,
                teg.optical_interfaces_to_node,
                teg.node_to_optical_interfaces
            )
            node_capacities = merge_many_node_capacities(node_capacities + scheduled_node_capacities)

            # Update the matrix containing the disabled contact time for state k
            W_dct += disabled_contact_time(teg.graphs[k], L_k, teg.state_durations[k])

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            ipn_node_to_planet_map=teg.ipn_node_to_planet_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )


class BruteForceScheduler:
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        scheduled_graphs = np.empty((teg.K, teg.N, teg.N), dtype='int64')
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")
        
        # See brute_force_matchings.py

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            ipn_node_to_planet_map=teg.ipn_node_to_planet_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )


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
        scheduled_graphs = np.zeros((teg.K, teg.N, teg.N), dtype='int64')
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        W_disabled_contact_time = np.zeros((teg.N, teg.N), dtype='int64')

        for k in tqdm(range(teg.K)):
            # Set the weights matrix equal to the current disabled contact time matrix
            weights[k] = W_disabled_contact_time

            # Compute max weight maximal matching using the blossom algorithm
            matched_edges = blossom(teg.graphs[k], weights[k])

            # Compute L_k from the matched edges
            L_k, contacts = build_graph(matched_edges, teg.graphs[k], teg.contacts[k], teg.node_map)
            scheduled_graphs[k] = L_k
            scheduled_contacts.append(contacts)

            # Update the matrix containing the disabled contact time for state k
            W_disabled_contact_time += disabled_contact_time(teg.graphs[k], L_k, teg.state_durations[k])

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            ipn_node_to_planet_map=teg.ipn_node_to_planet_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )


class RandomScheduler:
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        """
        Apply blossom algorithm with random weights
        """
        rng = np.random.default_rng(seed=42)

        # Since we are assigning the weights at random we have to do multiple iterations to avoid skewing the results
        # based on a single good or bad selection of weights. Generally 21 iterations is seen as statistically
        # significant.
        num_iters = 5

        all_scheduled_graphs = np.zeros((teg.K * num_iters, teg.N, teg.N), dtype='int64')
        scheduled_contacts = [[] for _ in range(teg.K)]
        all_weights = np.empty((teg.K * num_iters, teg.N, teg.N), dtype="int64")

        for i in range(num_iters):
            for k in tqdm(range(teg.K)):
                # Get an N x N matrix of weights randomly assigned between [0, 1], this will be used to compute the
                # matching.
                all_weights[k * i] = rng.integers(low=0, high=1, size=(teg.N, teg.N), endpoint=True)

                # Compute max weight maximal matching using the blossom algorithm but with the weights as a random
                # matrix. This gives the matching as if no real network information is known.
                matched_edges = blossom(teg.graphs[k], all_weights[k * i])

                # Compute L_k from the matched edges
                L_k, contacts = build_graph(matched_edges, teg.graphs[k], teg.contacts[k], teg.node_map)
                all_scheduled_graphs[k * i] = L_k

        scheduled_graphs = np.zeros((teg.K, teg.N, teg.N), dtype='int64')
        weights = np.empty((teg.K, teg.N, teg.N), dtype="int64")

        selected_ks = rng.choice(teg.K * num_iters, teg.K, replace=False)
        for k, selected_k in enumerate(selected_ks):
            scheduled_graphs[k] = all_scheduled_graphs[selected_k]
            weights[k] = all_weights[selected_k]

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            ipn_node_to_planet_map=teg.ipn_node_to_planet_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )


class AlternatingScheduler:
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        """
        The AlternatingScheduler is a naive algorithm that takes alternating turns between intra-constellation and
        inter-constellation transmissions. That is in the first state it will only schedule intra-constellation
        transmissions, then in the second state, only inter-constellation transmissions, and then repeat. There is also
        some randomness applied to the weights given in order to increase the fairness.
        """
        rng = np.random.default_rng(seed=42)

        scheduled_graphs = np.zeros((teg.K, teg.N, teg.N), dtype='int64')
        scheduled_contacts = []
        weights = np.zeros((teg.K, teg.N, teg.N), dtype="int64")

        for k in tqdm(range(teg.K)):
            # Set the weights for the maximal matching based on the alternating current state (even or odd) and based
            # on the transmission type (inter- or intra-constellation).
            for tx_idx in range(teg.N):
                for rx_idx in range(teg.N):
                    if teg.graphs[k][tx_idx][rx_idx] == 0:
                        continue

                    # Since these weights are just for fairness we don't need to do multiple iterations to converge on
                    # a result like the random algorithm
                    weight = rng.integers(low=0, high=10, size=1)[0]

                    # If it is an even state then assign the weights to the intra-constellation edges
                    is_intra_edge = tx_idx not in teg.ipn_node_to_planet_map and rx_idx in teg.ipn_node_to_planet_map
                    # If it is an odd state then assign the weights to the inter-constellation edges
                    is_inter_edge = tx_idx in teg.ipn_node_to_planet_map and rx_idx in teg.ipn_node_to_planet_map
                    weights[k][tx_idx][rx_idx] = \
                        weight if (k % 2 == 0 and is_intra_edge) or (k % 2 == 1 and is_inter_edge) else 0

            matched_edges = blossom(teg.graphs[k], weights[k])

            # Compute L_k from the matched edges
            L_k, contacts = build_graph(matched_edges, teg.graphs[k], teg.contacts[k], teg.node_map)
            scheduled_graphs[k] = L_k
            scheduled_contacts.append(contacts)

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            ipn_node_to_planet_map=teg.ipn_node_to_planet_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )

class OpticalTrunkLinkScheduler:
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        scheduled_graphs = np.empty((teg.K, teg.N, teg.N), dtype='int64')
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        node_capacities = []
        W_dct = np.zeros((teg.N, teg.N), dtype='int64')

        for k in tqdm(range(teg.K)):
            # Compute the change in network capacity on an edge by edge basis using the previous states node
            # capacities and the possible choices or decisions of active edges for this current state. This is a
            # dynamic programming approach where we used the memoized values of the weights of the previous k states
            # to compute the new weights matrix for capacity for state k+1.
            # We pass in the previous graphs as previous state, from there we can see for the two nodes making the edge
            # where were they looking before and where will they look now.
            # In the case that a node did not have a link in the previous time slice we can assume they are still
            # pointing at the last node they were in contact with.
            W_delta_cap = delta_capacity(
                teg.graphs[k],
                scheduled_graphs[:k],
                node_capacities,
                teg.nodes,
                teg.state_durations[k],
                teg.pos,
                teg.optical_interfaces_to_node,
                teg.node_to_optical_interfaces
            )

            # Compute the weight of each edge by doing a weighted sum of the capacity and fairness metrics
            weights[k] = ((1 - constants.alpha) * W_delta_cap) + (constants.alpha * W_dct)

            # Compute max weight maximal matching using the Hungarian algorithm
            matched_edges = max_weight_hungarian(teg.graphs[k], weights[k])

            # Compute L_k from the matched edges
            L_k, contacts = build_graph(matched_edges, teg.graphs[k], teg.contacts[k], teg.node_map)
            scheduled_graphs[k] = L_k
            scheduled_contacts.append(contacts)

            # Update node_capacities list with node capacities from state k contact plan and merge them together
            scheduled_node_capacities = compute_node_capacity_by_graph(
                L_k,
                teg.state_durations[k],
                teg.nodes,
                scheduled_graphs[:k],
                teg.pos,
                teg.optical_interfaces_to_node,
                teg.node_to_optical_interfaces
            )
            node_capacities = merge_many_node_capacities(node_capacities + scheduled_node_capacities)

            # Update the matrix containing the disabled contact time for state k
            W_dct += disabled_contact_time(teg.graphs[k], L_k, teg.state_durations[k])

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            ipn_node_to_planet_map=teg.ipn_node_to_planet_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )


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


def max_weight_hungarian(P_k: np.ndarray, W_k: np.ndarray) -> set:
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

    # The NetworkX hungarian algorithm impl uses a min weight maatching. To solve this as a max-weight we can apply the following
    # formula based on the max weight. max_weight + 1 - w
    max_weight = max(edge[2] for edge in edges)
    edges = [(edge[0], edge[1], max_weight + 1 - edge[2]) for edge in edges]

    # Create graph containing edges from P_k
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # Perform max weight matching. We leverage the networkx library to do this
    matched_edges = minimum_weight_full_matching(G)
    return {tuple(sorted((u, v))) for u, v in matched_edges.items() if u < v}


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
