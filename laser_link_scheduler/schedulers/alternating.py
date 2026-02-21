import numpy as np
from tqdm import tqdm
from laser_link_scheduler.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
)
from .base_scheduler import BaseScheduler


class AlternatingScheduler(BaseScheduler):
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
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
                    is_intra_edge = (
                        tx_idx not in teg.ipn_node_to_planet_map
                        and rx_idx in teg.ipn_node_to_planet_map
                    )
                    # If it is an odd state then assign the weights to the inter-constellation edges
                    is_inter_edge = (
                        tx_idx in teg.ipn_node_to_planet_map
                        and rx_idx in teg.ipn_node_to_planet_map
                    )
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
