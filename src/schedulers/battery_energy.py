import numpy as np

from src.constants import (
    MLConfig,
    OPTConfig,
    ALPHA,
)
from src.time_expanded_graph.time_expanded_graph import TimeExpandedGraph, Node
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

    def _project_generated_energy(
        self,
        from_time: int,
        to_time: int,
        initial_powers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rtg_generated_energy = generated_energy(
            from_time=from_time,
            to_time=to_time,
            initial_power=initial_powers,
            decay_constant=MLConfig.DECAY_RATE,
        )
        # NOTE: Implement the solar generation model later. For now the
        # battery model is pure RTG, so the battery is not recharged yet.
        solar_generated_energy = np.zeros_like(rtg_generated_energy)
        return rtg_generated_energy, solar_generated_energy

    def _weight_node_battery(
        self,
        n,
        state_duration: int,
        previous_schedule_contact_topology: np.ndarray,
        positions: np.ndarray,
        optical_interfaces_to_node: dict[int, int],
        nodes: list[Node],
        contact_topology_k: np.ndarray,
        generated_energy: tuple[np.ndarray, np.ndarray],
        battery_states: np.ndarray,
        oi_to_node_idx: np.ndarray,
        baseline_energy: np.ndarray,
    ) -> np.ndarray:
        weight_battery = np.zeros((n, n), dtype="float32")

        active_tx, active_rx = np.where(contact_topology_k >= 1)
        for tx_oi_idx, rx_oi_idx in zip(active_tx, active_rx):
            tx_node_idx = oi_to_node_idx[tx_oi_idx]
            rx_node_idx = oi_to_node_idx[rx_oi_idx]
            consumed_edge = transmission_energy(
                power=OPTConfig.AVG_TRANSMISSION_POWER,
                duration=compute_effective_contact_time(
                    oi_idx1=tx_oi_idx,
                    oi_idx2=rx_oi_idx,
                    scheduled_contact_topology=previous_schedule_contact_topology,
                    state_duration=state_duration,
                    positions=positions,
                    optical_interfaces_to_node=optical_interfaces_to_node,
                    nodes=nodes,
                    should_bypass_retargeting_time=self.should_bypass_retargeting_time,
                ),
            )
            # Still knowing that the battery recharge only with solar energy, we compute the difference considering the rtg energy
            # for admiting a difference between those nodes that can provide enought energy during the state and those who not.
            tx_projected_battery = (
                battery_states[tx_node_idx]
                + generated_energy[0][tx_node_idx]
                + generated_energy[1][tx_node_idx]
                - consumed_edge
                - baseline_energy[tx_node_idx]
            )
            rx_projected_battery = (
                battery_states[rx_node_idx]
                + generated_energy[0][rx_node_idx]
                + generated_energy[1][rx_node_idx]
                - consumed_edge
                - baseline_energy[rx_node_idx]
            )
            weight_battery[tx_oi_idx, rx_oi_idx] = min(
                tx_projected_battery,
                rx_projected_battery,
            )

        return weight_battery

    def _update_battery_states(
        self,
        state: int,
        teg: TimeExpandedGraph,
        adj_matrix: np.ndarray,
        previous_schedule_contact_topology: np.ndarray,
        battery_states: np.ndarray,
        generated_energy: tuple[np.ndarray, np.ndarray],
        baseline_energy: np.ndarray,
        oi_to_node_idx: np.ndarray,
        battery_max_capacity: np.ndarray,
    ) -> np.ndarray:
        total_generated_energy = generated_energy[0] + generated_energy[1]
        next_battery_states = battery_states.copy()
        state_duration = teg.state_durations[state]

        active_tx, active_rx = np.where(adj_matrix >= 1)
        for tx_oi_idx, rx_oi_idx in zip(active_tx, active_rx):
            tx_node_idx = oi_to_node_idx[tx_oi_idx]
            consumed_edge = transmission_energy(
                power=OPTConfig.AVG_TRANSMISSION_POWER,
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
            delta_energy = total_generated_energy[tx_node_idx] - (
                consumed_edge + baseline_energy[tx_node_idx]
            )
            if delta_energy < 0:
                # If need battery, substract from it
                next_battery_states[tx_node_idx] += delta_energy
            else:
                # Store remaining solar energy
                next_battery_states[tx_node_idx] += min(
                    delta_energy, generated_energy[0][tx_node_idx]
                )

        # Fix battery values to be within the (0, max_capacity) interval.
        finite_mask = np.isfinite(next_battery_states)
        next_battery_states[finite_mask] = np.clip(
            next_battery_states[finite_mask],
            0.0,
            battery_max_capacity[finite_mask],
        )
        return next_battery_states

    def schedule(
        self,
        teg: TimeExpandedGraph,
        progress_callback: ProgressCallback | None = None,
    ) -> TimeExpandedGraph:
        """
        Placeholder for battery energy schedule algorithm
        """
        n = teg.N
        k = teg.K
        scheduled_graphs = np.empty((k, n, n), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((k, n, n), dtype="float32")

        weights_dct = np.zeros((n, n), dtype="float32")
        accumulated_time: int = 0

        oi_to_node_idx = np.array(
            [teg.optical_interfaces_to_node[i] for i in range(n)],
            dtype=int,
        )
        node_ids = np.array(
            [node.id for node in teg.nodes],
            dtype=str,
        )

        initial_powers = MLConfig.get_initial_powers(node_ids).astype(
            "float64"
        )
        baseline_powers = MLConfig.get_baseline_powers(node_ids).astype(
            "float64"
        )
        battery_max_capacity = (
            MLConfig.get_initial_batteries(node_ids).astype("float64") * 3600.0
        )
        battery_states = battery_max_capacity.copy()
        for state in range(k):
            # Time restrictions for current state
            state_duration = teg.state_durations[state]
            from_time = accumulated_time
            to_time = accumulated_time + state_duration

            baseline_energy = baseline_powers * state_duration
            generated_energy = self._project_generated_energy(
                from_time,
                to_time,
                initial_powers,
            )
            weights_battery = self._weight_node_battery(
                n,
                state_duration,
                scheduled_graphs[:state],
                teg.pos,
                teg.optical_interfaces_to_node,
                teg.nodes,
                teg.graphs[state],
                generated_energy,
                battery_states,
                oi_to_node_idx,
                baseline_energy,
            )
            accumulated_time += state_duration
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
            battery_states = self._update_battery_states(
                state,
                teg,
                adj_matrix,
                scheduled_graphs[:state],
                battery_states,
                generated_energy,
                baseline_energy,
                oi_to_node_idx,
                battery_max_capacity,
            )

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
