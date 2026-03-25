import numpy as np

from src.constants import (
    MLConfig,
    OPTConfig,
    ALPHA,
    DESTINATION_NODES,
    RELAY_NODES,
    SOURCE_NODES,
)
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
from src.utils import ProgressCallback

from functools import total_ordering
from dataclasses import dataclass


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
        # node_bit_rate = OPTConfig.BIT_RATE

        initial_power = MLConfig.get_initial_power(node_id)
        decay_constant = MLConfig.DECAY_RATE
        minimum_power = transmission_energy(
            OPTConfig.AVG_TRANSMISSION_POWER, max_state_duration
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
        contact_topology_k: np.ndarray,
        node_lifespans: NetworkLifespan,
        accumulated_time: int,
        should_bypass_retargeting_time: bool,
    ) -> np.ndarray:
        weight = np.zeros((teg.N, teg.N), dtype="float32")

        state_duration = teg.state_durations[state]
        from_time = accumulated_time
        to_time = accumulated_time + state_duration

        oi_to_node_idx = np.array(
            [teg.optical_interfaces_to_node[i] for i in range(teg.N)],
            dtype=int,
        )
        oi_node_ids = np.array(
            [teg.nodes[node_idx].id for node_idx in oi_to_node_idx],
            dtype=str,
        )

        # TODO: add battery model
        initial_powers = MLConfig.get_initial_powers(oi_node_ids)
        generated_per_oi = generated_energy(
            from_time=from_time,
            to_time=to_time,
            initial_power=initial_powers,
            decay_constant=MLConfig.DECAY_RATE,
        )

        active_tx, active_rx = np.where(contact_topology_k >= 1)
        for tx_oi_idx, rx_oi_idx in zip(active_tx, active_rx):
            consumed_edge = transmission_energy(
                power=OPTConfig.PEAK_TRANSMISSION_POWER,
                duration=compute_effective_contact_time(
                    oi_idx1=tx_oi_idx,
                    oi_idx2=rx_oi_idx,
                    scheduled_contact_topology=previous_schedule_contact_topology,
                    state_duration=state_duration,
                    positions=teg.pos,
                    optical_interfaces_to_node=teg.optical_interfaces_to_node,
                    nodes=teg.nodes,
                    should_bypass_retargeting_time=should_bypass_retargeting_time,
                ),
            )
            min_generated_edge = min(
                generated_per_oi[tx_oi_idx],
                generated_per_oi[rx_oi_idx],
            )
            weight[tx_oi_idx, rx_oi_idx] = min_generated_edge - consumed_edge

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

    def schedule(
        self,
        teg: TimeExpandedGraph,
        progress_callback: ProgressCallback | None = None,
    ) -> TimeExpandedGraph:
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
        accumulated_time = 0
        for state in range(teg.K):
            weights_lifespan = self._weight_node_lifespans(
                state,
                scheduled_graphs[:state],
                teg,
                teg.graphs[state],
                node_lifespans,
                accumulated_time,
                False,
            )
            accumulated_time += teg.state_durations[state]

            # Compute the weight of each edge by doing a weighted sum of the lifespan and fairness metrics
            weights[state] = ((1 - ALPHA) * weights_lifespan) + (
                ALPHA * weights_dct
            )

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
            # For the percentage on running table
            if progress_callback is not None:
                progress_callback("schedule", state + 1, teg.K)

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
