import networkx as nx
import numpy as np
from tqdm import tqdm

from laser_link_scheduler import constants
from laser_link_scheduler.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
)
from .base_scheduler import BaseScheduler
from laser_link_scheduler.topology.contact_plan import Contact
from laser_link_scheduler.topology.weights import (
    compute_node_capacity_by_graph,
    delta_capacity,
    disabled_contact_time,
    merge_many_node_capacities,
)


class RandomScheduler(BaseScheduler):
    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        """
        Apply blossom algorithm with random weights
        """
        rng = np.random.default_rng(seed=42)

        # Since we are assigning the weights at random we have to do multiple iterations to avoid skewing the results
        # based on a single good or bad selection of weights. Generally 21 iterations is seen as statistically
        # significant.
        num_iters = 5

        all_scheduled_graphs = np.zeros(
            (teg.K * num_iters, teg.N, teg.N), dtype="int64"
        )
        scheduled_contacts = [[] for _ in range(teg.K)]
        all_weights = np.empty(
            (teg.K * num_iters, teg.N, teg.N), dtype="int64"
        )

        for i in range(num_iters):
            for k in tqdm(range(teg.K)):
                # Get an N x N matrix of weights randomly assigned between [0, 1], this will be used to compute the
                # matching.
                all_weights[k * i] = rng.integers(
                    low=0, high=1, size=(teg.N, teg.N), endpoint=True
                )

                # Compute max weight maximal matching using the blossom algorithm but with the weights as a random
                # matrix. This gives the matching as if no real network information is known.
                matched_edges = self._blossom(
                    teg.graphs[k], all_weights[k * i]
                )

                # Compute L_k from the matched edges
                L_k, contacts = self._build_graph(
                    matched_edges, teg.graphs[k], teg.contacts[k], teg.node_map
                )
                all_scheduled_graphs[k * i] = L_k

        scheduled_graphs = np.zeros((teg.K, teg.N, teg.N), dtype="int64")
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
