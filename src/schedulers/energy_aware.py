import numpy as np

from src.constants import (
    MLConfig,
    OPTConfig,
    ALPHA,
)
from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
    Node,
)
from .base_scheduler import BaseScheduler

from src.topology.weights import (
    disabled_contact_time,
    compute_effective_contact_time,
)
from src.models import *
from src.utils import ProgressCallback


class EnergyAware(BaseScheduler):
    def __init__(self, should_bypass_retargeting_time=False):
        super().__init__(should_bypass_retargeting_time)

    def _weight_node_energy(
        self,
        n: int,
        state_duration: int,
        previous_schedule_contact_topology: np.ndarray,
        teg: TimeExpandedGraph,
        contacts_for_current_state: np.ndarray,
        optical_interfaces_to_node: dict[int, int],
        nodes: list[Node],
        generated_per_oi: np.ndarray,
    ) -> np.ndarray:
        weight_energy = np.zeros((n, n), dtype="float32")

        active_tx, active_rx = np.where(contacts_for_current_state >= 1)
        for tx_oi_idx, rx_oi_idx in zip(active_tx, active_rx):
            consumed_edge = transmission_energy(
                power=OPTConfig.AVG_TRANSMISSION_POWER,
                duration=compute_effective_contact_time(
                    oi_idx1=tx_oi_idx,
                    oi_idx2=rx_oi_idx,
                    scheduled_contact_topology=previous_schedule_contact_topology,
                    state_duration=state_duration,
                    positions=teg.pos,
                    optical_interfaces_to_node=optical_interfaces_to_node,
                    nodes=nodes,
                    should_bypass_retargeting_time=self.should_bypass_retargeting_time,
                ),
            )
            min_generated_edge = min(
                generated_per_oi[tx_oi_idx],
                generated_per_oi[rx_oi_idx],
            )
            weight_energy[tx_oi_idx, rx_oi_idx] = (
                min_generated_edge - consumed_edge
            )

        return weight_energy

    def schedule(
        self,
        teg: TimeExpandedGraph,
        progress_callback: ProgressCallback | None = None,
    ) -> TimeExpandedGraph:
        """
        Placeholder for energy schedule algorithm
        """
        n = teg.N
        k = teg.K

        scheduled_graphs = np.empty((k, n, n), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((k, n, n), dtype="float32")

        weights_dct = np.zeros((n, n), dtype="float32")
        accumulated_time: int = 0

        nodes = teg.nodes
        oi_to_node_idx = np.array(
            [teg.optical_interfaces_to_node[i] for i in range(n)],
            dtype=int,
        )
        oi_node_ids = np.array(
            [nodes[node_idx].id for node_idx in oi_to_node_idx],
            dtype=str,
        )

        initial_powers = MLConfig.get_initial_powers(oi_node_ids)
        state_contacts_graphs = teg.graphs
        for state in range(k):
            # Time restrictions
            state_duration = teg.state_durations[state]
            from_time = accumulated_time
            to_time = from_time + state_duration

            # Generated energy per optical interface
            generated_per_oi = generated_energy(
                from_time=from_time,
                to_time=to_time,
                initial_power=initial_powers,
                decay_constant=MLConfig.DECAY_RATE,
            )

            weights_energy = self._weight_node_energy(
                n,
                state_duration,
                scheduled_graphs[:state],
                teg,
                state_contacts_graphs[state],
                teg.optical_interfaces_to_node,
                nodes,
                generated_per_oi,
            )
            accumulated_time += state_duration
            # Compute the weight of each edge by doing a weighted sum of the lifespan and fairness metrics
            weights[state] = ((1 - ALPHA) * weights_energy) + (
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
            # Update the matrix containing the disabled contact time for current state
            weights_dct += disabled_contact_time(
                teg.graphs[state], adj_matrix, teg.state_durations[state]
            )
            # For the percentage on running table
            if progress_callback is not None:
                progress_callback("schedule", state + 1, k)

        return TimeExpandedGraph(
            graphs=scheduled_graphs,
            contacts=scheduled_contacts,
            state_durations=teg.state_durations,
            K=k,
            N=n,
            nodes=teg.nodes,
            node_map=teg.node_map,
            W=weights,
            pos=teg.pos,
            optical_interfaces_to_node=teg.optical_interfaces_to_node,
            node_to_optical_interfaces=teg.node_to_optical_interfaces,
            effective_contact_durations=teg.effective_contact_durations,
        )
