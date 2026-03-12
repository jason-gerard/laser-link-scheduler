import numpy as np
from tqdm import tqdm

from src import constants
from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
    Node,
)
from .base_scheduler import BaseScheduler
from src.topology.contact_plan import Contact
from src.topology.weights import (
    disabled_contact_time,
    compute_effective_contact_time,
)
from src.models import *

from functools import total_ordering
from dataclasses import dataclass

MLcfg = constants.MLConfig
OPTcfg = constants.OPTConfig


@dataclass
@total_ordering
class NodeLifespan:
    id: str
    mission_lifespan: float

    def __eq__(self, value):
        if not isinstance(value, NodeLifespan):
            return self.mission_lifespan == value
        return self.mission_lifespan == value.mission_lifespan

    def __lt__(self, value):
        if not isinstance(value, NodeLifespan):
            return self.mission_lifespan < value
        return self.mission_lifespan < value.mission_lifespan


# Map each node lifespan
# NOTE: Review if it's worth it.
NetworkLifespan = dict[str, NodeLifespan]


class LifespanAware(BaseScheduler):
    def _get_node_lifespan(
        self, node_id: str, max_state_duration: int
    ) -> NodeLifespan:
        # TODO: Implement with bit_rate() from models, to obtain a dinamic bit_rate
        # node_bit_rate = OPTcfg.BIT_RATE

        initial_power = MLcfg.get_initial_power(node_id)
        decay_constant = MLcfg.DECAY_RATE
        minimum_power = transmission_energy(
            OPTcfg.AVG_TRANSMISSION_POWER, max_state_duration
        )

        return NodeLifespan(
            id=node_id,
            mission_lifespan=mission_lifetime(
                initial_power, decay_constant, minimum_power
            ),
        )

    def _get_network_lifespan(
        self, nodes: list[Node], max_state_duration: int
    ) -> NetworkLifespan:
        return {
            node.id: self._get_node_lifespan(node.id, max_state_duration)
            for node in nodes
        }

    def _weight_node_lifespans(
        self,
        state: int,
        previous_schedule_contact_topology: np.ndarray,
        teg: TimeExpandedGraph,
        node_lifespans: NetworkLifespan,
    ) -> np.ndarray:
        weight = np.zeros((teg.N, teg.N), dtype="float32")

        for tx_oi_idx in range(teg.N):
            for rx_oi_idx in range(teg.N):
                # Skip if not contact
                if not teg.graphs[state][tx_oi_idx][rx_oi_idx]:
                    continue

                # TODO: Change to make it more efficient.
                tx_node_lifespan = node_lifespans[
                    teg.nodes[teg.optical_interfaces_to_node[tx_oi_idx]].id
                ]
                rx_node_lifespan = node_lifespans[
                    teg.nodes[teg.optical_interfaces_to_node[rx_oi_idx]].id
                ]
                # TODO: Adapt to use the node unique bit rate.
                #       This change implies changes on lib's arquitecture
                bit_rate = OPTcfg.BIT_RATE

                # TODO: consider the real
                #       In this first approach we'll bypass the retargeting time
                #       (effective_contact_duration = state_duration)
                effective_contact_duration = compute_effective_contact_time(
                    tx_oi_idx,
                    rx_oi_idx,
                    previous_schedule_contact_topology,
                    teg.state_durations[state],
                    teg.pos,
                    teg.optical_interfaces_to_node,
                    teg.nodes,
                    should_bypass_retargeting_time=True,
                )

                # Edge's weight define as minimum lifespan between nodes
                weight[tx_oi_idx][rx_oi_idx] = min(
                    tx_node_lifespan, rx_node_lifespan
                )

                # TODO: Consider the maximum between the minimun power usage within the two satellites
                #       As a checker, if the weight is lower the is set as 0.
                #       We use here the eff_contact_duration and bit_rate, and anything else.
        return weight

    # def _update_node_lifespans(
    #     self,
    #     state: int,
    #     L_k: np.ndarray,
    #     state_duration: int,
    #     nodes: list[Node],
    #     scheduled_graphs: np.ndarray,
    #     weights_delta_lfspn: np.ndarray,
    # ) -> list[NodeLifespan]:
    #     new_node_lifespan: list[NodeLifespan] = []

    #     # For each contact calculate the energy transmition consumption

    #     return new_node_lifespan

    def schedule(self, teg: TimeExpandedGraph) -> TimeExpandedGraph:
        """
        Placeholder for lifespan schedule algorithm
        """
        scheduled_graphs = np.empty((teg.K, teg.N, teg.N), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        node_lifespans: NetworkLifespan = self._get_network_lifespan(
            teg.nodes, teg.max_state_duration
        )
        weights_dct = np.zeros((teg.N, teg.N), dtype="float32")

        for state in tqdm(range(teg.K)):
            #
            # Description here
            #
            weights_delta_lifespan = self._weight_node_lifespans(
                state, scheduled_graphs[:state], teg, node_lifespans
            )

            # Compute the weight of each edge by doing a weighted sum of the lifespan and fairness metrics
            weights[state] = (
                (1 - constants.alpha) * weights_delta_lifespan
            ) + (constants.alpha * weights_dct)

            # Compute max weight maximal matching using the blossom algorithm
            matched_edges = self._blossom(teg.graphs[state], weights[state])

            # Compute L_k from the matched edges
            adj_matrix, contacts = self._build_graph(
                matched_edges,
                teg.graphs[state],
                teg.contacts[state],
                teg.node_map,
            )
            scheduled_graphs[state] = adj_matrix
            scheduled_contacts.append(contacts)

            # TODO: Make the function haha
            # Update node_lifespan list with node lifespans from current state contact plan and merge them together
            # node_lifespans = self._update_node_lifespans(
            #     adj_matrix,
            #     teg.state_durations[state],
            #     teg.nodes,
            #     scheduled_graphs[:state],
            #     weights_delta_lifespan,
            # )

            # Update the matrix containing the disabled contact time for current state
            weights_dct += disabled_contact_time(
                teg.graphs[state], adj_matrix, teg.state_durations[state]
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
