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
    disabled_contact_time,
    merge_many_node_capacities,
)

from functools import total_ordering


@total_ordering
class NodeLifespan:
    def __init__(self, id: str, mission_lifespan: float):
        self.id = id
        self.mission_lifespan = mission_lifespan

    def __eq__(self, value):
        if not isinstance(value, NodeLifespan):
            return self.mission_lifespan == value
        return self.mission_lifespan == value.mission_lifespan

    def __lt__(self, value):
        if not isinstance(value, NodeLifespan):
            return self.mission_lifespan < value
        return self.mission_lifespan < value.mission_lifespan


class LifespanAware(BaseScheduler):
    def _delta_lifespan(
        self,
        contact_topology_k: np.ndarray,
        node_lifespans: list[NodeLifespan],
    ) -> np.ndarray:
        num_nodes = len(contact_topology_k)
        delta_lifespans = np.zeros((num_nodes, num_nodes), dtype="float32")
        for tx_idx in range(num_nodes):
            for rx_idx in range(num_nodes):
                if contact_topology_k[tx_idx][rx_idx]:
                    delta_lifespans[tx_idx][rx_idx] = min(
                        node_lifespans[tx_idx], node_lifespans[rx_idx]
                    )

        return delta_lifespans

    def _compute_node_lifespan(
        self,
        L_k: np.ndarray,
        state_duration: int,
        nodes: list[str],
        scheduled_graphs: np.ndarray,
        weights_delta_lfspn: np.ndarray,
    ) -> list[NodeLifespan]:
        new_node_lifespan = []
        for node in nodes:
            new_node_lifespan.append(
                NodeLifespan(
                    id=node,
                    mission_lifespan=(
                        weights_delta_lfspn[node][node] - state_duration
                    ),
                )
            )
        return new_node_lifespan

    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        """
        Placeholder for lifespan schedule algorithm
        """
        scheduled_graphs = np.empty((teg.K, teg.N, teg.N), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        node_lifespans = []
        weights_dct = np.zeros((teg.N, teg.N), dtype="int64")

        for state in tqdm(range(teg.K)):
            #
            # text here
            #
            weights_delta_lfspn = self._delta_lifespan(
                teg.graphs[state], node_lifespans
            )

            # Compute the weight of each edge by doing a weighted sum of the lifespan and fairness metrics
            weights[state] = ((1 - constants.alpha) * weights_delta_lfspn) + (
                constants.alpha * weights_dct
            )

            # Compute max weight maximal matching using the blossom algorithm
            matched_edges = self._blossom(teg.graphs[state], weights[state])

            # Compute L_k from the matched edges
            L_k, contacts = self._build_graph(
                matched_edges,
                teg.graphs[state],
                teg.contacts[state],
                teg.node_map,
            )
            scheduled_graphs[state] = L_k
            scheduled_contacts.append(contacts)

            # Update node_lifespan list with node lifespans from current state contact plan and merge them together
            scheduled_node_lifespans = self._compute_node_lifespan(
                L_k,
                teg.state_durations[state],
                teg.nodes,
                scheduled_graphs[:state],
                weights_delta_lfspn,
            )

            # Update the matrix containing the disabled contact time for current state
            weights_dct += disabled_contact_time(
                teg.graphs[state], L_k, teg.state_durations[state]
            )

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
