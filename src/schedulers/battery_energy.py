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


class BatteryEnergy(BaseScheduler):
    def __init__(self, should_bypass_retargeting_time=False):
        super().__init__(should_bypass_retargeting_time)
        self.battery_states: np.ndarray | None = None

    def _weight_node_battery(
        self,
        state: int,
        previous_schedule_contact_topology: np.ndarray,
        teg: TimeExpandedGraph,
        contact_topology_k: np.ndarray,
        accumulated_time: int,
        battery_states: np.ndarray,
        initial_powers: np.ndarray,
    ) -> np.ndarray:
        state_duration = teg.state_durations[state]
        from_time = accumulated_time
        to_time = accumulated_time + state_duration

        weight_battery = np.zeros((teg.N, teg.N), dtype="float32")
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
                    should_bypass_retargeting_time=self.should_bypass_retargeting_time,
                ),
            )
            min_generated_edge = min(
                generated_per_oi[tx_oi_idx],
                generated_per_oi[rx_oi_idx],
            )
            delta_energy = min_generated_edge - consumed_edge

            weight_battery[tx_oi_idx, rx_oi_idx] = min(
                battery_states[tx_oi_idx], battery_states[rx_oi_idx]
            )
            # There are two ways to do it:
            #
            # Modify the weight only if the delta_energy is below 0.
            # That modelate the discharge of the battery
            if delta_energy < 0:
                weight_battery[tx_oi_idx, rx_oi_idx] += delta_energy

            # Modify the weight allways
            # This provoque more difference bewteen those nodes that not provide enoght energy to supply the consumption and
            # those who can provide energy. In the other case the weight only depends by the battery here we add the energy in the game
            # weight_battery[tx_oi_idx, rx_oi_idx] += delta_energy

        return weight_battery

    # def _update_battery_states(
    #     battery_states: np.ndarray,
    #     adj_matrix: np.ndarray,
    # ) -> np.ndarray: ...

    def schedule(
        self,
        teg: TimeExpandedGraph,
        progress_callback: ProgressCallback | None = None,
    ) -> TimeExpandedGraph:
        """
        Placeholder for battery energy schedule algorithm
        """
        scheduled_graphs = np.empty((teg.K, teg.N, teg.N), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        weights_dct = np.zeros((teg.N, teg.N), dtype="float32")
        accumulated_time = 0

        oi_to_node_idx = np.array(
            [teg.optical_interfaces_to_node[i] for i in range(teg.N)],
            dtype=int,
        )
        oi_node_ids = np.array(
            [teg.nodes[node_idx].id for node_idx in oi_to_node_idx],
            dtype=str,
        )

        battery_states = MLConfig.get_initial_batteries(oi_node_ids)
        initial_powers = MLConfig.get_initial_powers(oi_node_ids)

        for state in range(teg.K):
            weights_battery = self._weight_node_battery(
                state,
                scheduled_graphs[:state],
                teg,
                teg.graphs[state],
                accumulated_time,
                battery_states,
                initial_powers,
            )
            accumulated_time += teg.state_durations[state]
            # Compute the weight of each edge by doing a weighted sum of the lifespan and fairness metrics
            weights[state] = ((1 - ALPHA) * weights_battery) + (
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

            # Update the battery states
            # battery_states = self._update_battery_states(
            #     battery_states,
            #     adj_matrix,
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
