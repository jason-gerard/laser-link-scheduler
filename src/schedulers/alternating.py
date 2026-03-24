import numpy as np
from src.constants import RELAY_NODES
from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
)
from .base_scheduler import BaseScheduler
from src.utils import ProgressCallback


class AlternatingScheduler(BaseScheduler):
    def schedule(
        self,
        teg: TimeExpandedGraph,
        progress_callback: ProgressCallback | None = None,
    ) -> TimeExpandedGraph:
        """
        The AlternatingScheduler is a naive algorithm that takes alternating turns between intra-constellation and
        inter-constellation transmissions. That is in the first state it will only schedule intra-constellation
        transmissions, then in the second state, only inter-constellation transmissions, and then repeat. There is also
        some randomness applied to the weights given in order to increase the fairness.
        """
        rng = np.random.default_rng(seed=42)

        scheduled_graphs = np.zeros((teg.K, teg.N, teg.N), dtype="int64")
        scheduled_contacts = []
        weights = np.zeros((teg.K, teg.N, teg.N), dtype="int64")

        for k in range(teg.K):
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
                    tx_node = teg.nodes[teg.optical_interfaces_to_node[tx_idx]]
                    rx_node = teg.nodes[teg.optical_interfaces_to_node[rx_idx]]
                    tx_is_ipn = tx_node.id in RELAY_NODES
                    rx_is_ipn = rx_node.id in RELAY_NODES
                    is_intra_edge = not tx_is_ipn and rx_is_ipn
                    # If it is an odd state then assign the weights to the inter-constellation edges
                    is_inter_edge = tx_is_ipn and rx_is_ipn
                    weights[k][tx_idx][rx_idx] = (
                        weight
                        if (k % 2 == 0 and is_intra_edge)
                        or (k % 2 == 1 and is_inter_edge)
                        else 0
                    )

            matched_edges = self._blossom(teg.graphs[k], weights[k])

            # Compute L_k from the matched edges
            L_k, contacts = self._build_graph(
                matched_edges, teg.graphs[k], teg.contacts[k], teg.node_map
            )
            scheduled_graphs[k] = L_k
            scheduled_contacts.append(contacts)
            # For the percentage on running table
            if progress_callback is not None:
                progress_callback("schedule", k + 1, teg.K)

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
