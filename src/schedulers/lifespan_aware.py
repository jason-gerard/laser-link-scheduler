from dataclasses import dataclass
from functools import total_ordering

import numpy as np

from src.constants import ALPHA, MLConfig, OCTConfig
from src.models import generating_power, generated_energy, mission_lifetime
from src.models.energy_comsumption import transmission_energy
from src.time_expanded_graph.time_expanded_graph import Node, TimeExpandedGraph
from src.topology.weights import (
    compute_effective_contact_time,
    disabled_contact_time,
)
from src.utils import ProgressCallback

from .base_scheduler import BaseScheduler


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


NetworkLifespan = dict[str, NodeLifespan]


class LifespanAware(BaseScheduler):
    def __init__(self, should_bypass_retargeting_time: bool = False):
        super().__init__(should_bypass_retargeting_time)
        self.battery_states: np.ndarray | None = None
        self.node_lifespans: NetworkLifespan = {}

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

    def _estimate_remaining_lifetime(
        self,
        node_id: str,
        current_time: int,
        remaining_battery_energy: float,
        required_power: float,
    ) -> float:
        if not np.isfinite(remaining_battery_energy):
            return float("inf")
        if required_power <= 0:
            return float("inf")

        initial_power = MLConfig.get_initial_power(node_id)
        if not np.isfinite(initial_power):
            return float("inf")

        lam = MLConfig.DECAY_RATE

        total_rtg_lifetime = mission_lifetime(
            initial_power,
            lam,
            required_power,
        )
        remaining_rtg_lifetime = max(total_rtg_lifetime - current_time, 0.0)
        battery_buffer_lifetime = remaining_battery_energy / required_power
        return remaining_rtg_lifetime + battery_buffer_lifetime

    def _weight_node_lifespans(
        self,
        n: int,
        state_duration: int,
        previous_schedule_contact_topology: np.ndarray,
        positions: np.ndarray,
        optical_interfaces_to_node: dict[int, int],
        nodes: list[Node],
        contact_topology_k: np.ndarray,
        generated_energy_by_source: tuple[np.ndarray, np.ndarray],
        battery_states: np.ndarray,
        oi_to_node_idx: np.ndarray,
        baseline_energy: np.ndarray,
        baseline_powers: np.ndarray,
        current_time: int,
    ) -> np.ndarray:
        weight_lifespan = np.zeros((n, n), dtype="float32")

        active_tx, active_rx = np.where(contact_topology_k >= 1)
        for tx_oi_idx, rx_oi_idx in zip(active_tx, active_rx):
            tx_node_idx = oi_to_node_idx[tx_oi_idx]
            rx_node_idx = oi_to_node_idx[rx_oi_idx]
            consumed_edge = transmission_energy(
                power=OCTConfig.AVG_TRANSMISSION_POWER,
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

            tx_projected_battery = (
                battery_states[tx_node_idx]
                + generated_energy_by_source[1][tx_node_idx]
                - consumed_edge
                - baseline_energy[tx_node_idx]
            )
            rx_projected_battery = (
                battery_states[rx_node_idx]
                + generated_energy_by_source[1][rx_node_idx]
                - consumed_edge
                - baseline_energy[rx_node_idx]
            )

            tx_required_power = baseline_powers[tx_node_idx] + (
                consumed_edge / max(state_duration, 1)
            )
            rx_required_power = baseline_powers[rx_node_idx] + (
                consumed_edge / max(state_duration, 1)
            )

            tx_remaining_lifetime = self._estimate_remaining_lifetime(
                nodes[tx_node_idx].id,
                current_time + state_duration,
                max(tx_projected_battery, 0.0),
                tx_required_power,
            )
            rx_remaining_lifetime = self._estimate_remaining_lifetime(
                nodes[rx_node_idx].id,
                current_time + state_duration,
                max(rx_projected_battery, 0.0),
                rx_required_power,
            )

            weight_lifespan[tx_oi_idx, rx_oi_idx] = min(
                tx_remaining_lifetime,
                rx_remaining_lifetime,
            )

        return weight_lifespan

    def _update_battery_states(
        self,
        state: int,
        teg: TimeExpandedGraph,
        adj_matrix: np.ndarray,
        previous_schedule_contact_topology: np.ndarray,
        battery_states: np.ndarray,
        generated_energy_by_source: tuple[np.ndarray, np.ndarray],
        baseline_energy: np.ndarray,
        oi_to_node_idx: np.ndarray,
        battery_max_capacity: np.ndarray,
    ) -> np.ndarray:
        total_generated_energy = (
            generated_energy_by_source[0] + generated_energy_by_source[1]
        )
        next_battery_states = battery_states.copy()
        state_duration = teg.state_durations[state]

        active_tx, active_rx = np.where(adj_matrix >= 1)
        for tx_oi_idx, rx_oi_idx in zip(active_tx, active_rx):
            tx_node_idx = oi_to_node_idx[tx_oi_idx]
            consumed_edge = transmission_energy(
                power=OCTConfig.AVG_TRANSMISSION_POWER,
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
                next_battery_states[tx_node_idx] += delta_energy
            else:
                next_battery_states[tx_node_idx] += min(
                    delta_energy,
                    generated_energy_by_source[1][tx_node_idx],
                )

        finite_mask = np.isfinite(next_battery_states)
        next_battery_states[finite_mask] = np.clip(
            next_battery_states[finite_mask],
            0.0,
            battery_max_capacity[finite_mask],
        )
        return next_battery_states

    def _compute_network_lifespans(
        self,
        nodes: list[Node],
        battery_states: np.ndarray,
        baseline_powers: np.ndarray,
        current_time: int,
    ) -> NetworkLifespan:
        lifespans: NetworkLifespan = {}
        for node_idx, node in enumerate(nodes):
            lifespans[node.id] = NodeLifespan(
                id=node.id,
                mission_lifespan=self._estimate_remaining_lifetime(
                    node.id,
                    current_time,
                    battery_states[node_idx],
                    baseline_powers[node_idx],
                ),
            )
        return lifespans

    def schedule(
        self,
        teg: TimeExpandedGraph,
        progress_callback: ProgressCallback | None = None,
    ) -> TimeExpandedGraph:
        scheduled_graphs = np.empty((teg.K, teg.N, teg.N), dtype="int64")
        scheduled_contacts = []
        weights = np.empty((teg.K, teg.N, teg.N), dtype="float32")

        n = teg.N
        k = teg.K
        weights_dct = np.zeros((n, n), dtype="float32")
        accumulated_time = 0

        oi_to_node_idx = np.array(
            [teg.optical_interfaces_to_node[i] for i in range(n)],
            dtype=int,
        )
        node_ids = np.array([node.id for node in teg.nodes], dtype=str)

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
            state_duration = teg.state_durations[state]
            from_time = accumulated_time
            to_time = accumulated_time + state_duration

            baseline_energy = baseline_powers * state_duration
            generated_energy_by_source = self._project_generated_energy(
                from_time,
                to_time,
                initial_powers,
            )
            weights_lifespan = self._weight_node_lifespans(
                n,
                state_duration,
                scheduled_graphs[:state],
                teg.pos,
                teg.optical_interfaces_to_node,
                teg.nodes,
                teg.graphs[state],
                generated_energy_by_source,
                battery_states,
                oi_to_node_idx,
                baseline_energy,
                baseline_powers,
                accumulated_time,
            )
            accumulated_time += state_duration
            weights[state] = ((1 - ALPHA) * weights_lifespan) + (
                ALPHA * weights_dct
            )

            matched_edges = self._blossom(teg.graphs[state], weights[state])
            adj_matrix, contacts = self._build_graph(
                matched_edges,
                teg.graphs[state],
                teg.contacts[state],
                teg.node_map,
            )
            scheduled_graphs[state] = adj_matrix
            scheduled_contacts.append(contacts)

            battery_states = self._update_battery_states(
                state,
                teg,
                adj_matrix,
                scheduled_graphs[:state],
                battery_states,
                generated_energy_by_source,
                baseline_energy,
                oi_to_node_idx,
                battery_max_capacity,
            )
            self.node_lifespans = self._compute_network_lifespans(
                teg.nodes,
                battery_states,
                baseline_powers,
                accumulated_time,
            )

            weights_dct += disabled_contact_time(
                teg.graphs[state], adj_matrix, teg.state_durations[state]
            )
            if progress_callback is not None:
                progress_callback("schedule", state + 1, k)

        self.battery_states = battery_states

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
