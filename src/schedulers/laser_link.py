import networkx as nx
import numpy as np
from tqdm import tqdm

from src import constants
from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
)
from .base_scheduler import BaseScheduler
from src.topology.weights import (
    compute_node_capacity_by_graph,
    delta_capacity,
    disabled_contact_time,
    merge_many_node_capacities,
)


class LaserLinkScheduler(BaseScheduler):
    def __init__(self, should_bypass_retargeting_time=False):
        super().__init__()
        self.should_bypass_retargeting_time: bool = (
            should_bypass_retargeting_time
        )

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
        scheduled_graphs = np.empty((teg.K, teg.N, teg.N), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        node_capacities = []
        W_dct = np.zeros((teg.N, teg.N), dtype="int64")

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
                self.should_bypass_retargeting_time,
            )

            # Compute the weight of each edge by doing a weighted sum of the capacity and fairness metrics
            weights[k] = ((1 - constants.alpha) * W_delta_cap) + (
                constants.alpha * W_dct
            )

            # Compute max weight maximal matching using the blossom algorithm
            matched_edges = self._blossom(teg.graphs[k], weights[k])

            # Compute L_k from the matched edges
            L_k, contacts = self._build_graph(
                matched_edges, teg.graphs[k], teg.contacts[k], teg.node_map
            )
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
                teg.node_to_optical_interfaces,
                self.should_bypass_retargeting_time,
            )
            node_capacities = merge_many_node_capacities(
                node_capacities + scheduled_node_capacities
            )

            # Update the matrix containing the disabled contact time for state k
            W_dct += disabled_contact_time(
                teg.graphs[k], L_k, teg.state_durations[k]
            )

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=teg.K,
            N=teg.N,
            nodes=teg.nodes,
            node_map=teg.node_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )
