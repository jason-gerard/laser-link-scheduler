import numpy as np
from tqdm import tqdm

from laser_link_scheduler.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
)
from .base_scheduler import BaseScheduler
from laser_link_scheduler.topology.weights import (
    disabled_contact_time,
)


class FairContactPlan(BaseScheduler):
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
        scheduled_graphs = np.zeros((teg.K, teg.N, teg.N), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        W_disabled_contact_time = np.zeros((teg.N, teg.N), dtype="int64")

        for k in tqdm(range(teg.K)):
            # Set the weights matrix equal to the current disabled contact time matrix
            weights[k] = W_disabled_contact_time

            # Compute max weight maximal matching using the blossom algorithm
            matched_edges = self._blossom(teg.graphs[k], weights[k])

            # Compute L_k from the matched edges
            L_k, contacts = self._build_graph(
                matched_edges, teg.graphs[k], teg.contacts[k], teg.node_map
            )
            scheduled_graphs[k] = L_k
            scheduled_contacts.append(contacts)

            # Update the matrix containing the disabled contact time for state k
            W_disabled_contact_time += disabled_contact_time(
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
