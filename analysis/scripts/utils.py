import math
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.constants import OPTConfig, MLConfig
from src.models import (
    mission_lifetime,
    generating_power,
    transmission_energy,
    transmission_duration,
    bit_rate,
    generated_energy,
)
from src.topology.weights import compute_effective_contact_time
from src.time_expanded_graph import TimeExpandedGraph


def compute_energy_metrics(
    teg: TimeExpandedGraph,
    tx_oi_idx: int,
    rx_oi_idx: int,
    state_contact_topology: Any,
    state_duration: int,
    accumulated_time: float,
    tx_node_id: str,
    should_bypass_retargeting_time: bool,
):
    from_time = accumulated_time
    to_time = accumulated_time + state_duration
    initial_power = MLConfig.get_initial_power(tx_node_id)

    # Obtain the generated and consumed energy during the state duration
    generated: float = _get_generated_energy(
        from_time, to_time, initial_power
    )  # Joules
    consumed: float = _get_consumed_energy(
        teg,
        tx_oi_idx,
        rx_oi_idx,
        state_contact_topology,
        state_duration,
        should_bypass_retargeting_time,
    )  # Joules

    return generated, consumed


def _get_generated_energy(from_time, to_time, initial_power):
    # Assuming that the power generation follows the RTG model with exponential decay. Then, we can compute the generated energy during the state duration as the integral of the power function over the state duration.
    # And the conversion efficiency is 100% for simplicity.
    decay_constant = MLConfig.DECAY_RATE
    energy = generated_energy(
        from_time=from_time,
        to_time=to_time,
        initial_power=initial_power,
        decay_constant=decay_constant,
    )
    return energy


def _get_consumed_energy(
    teg: TimeExpandedGraph,
    tx_oi_idx: int,
    rx_oi_idx: int,
    state_contact_topology: Any,
    state_duration: int,
    should_bypass_retargeting_time: bool,
):
    # Obtain the effective contact time
    effective_contact_time = compute_effective_contact_time(
        tx_oi_idx1=tx_oi_idx,
        rx_oi_idx2=rx_oi_idx,
        scheduled_contact_topology=state_contact_topology,
        state_duration=state_duration,
        positions=teg.pos,
        optical_interfaces_to_node=teg.optical_interfaces_to_node,
        nodes=teg.nodes,
        should_bypass_retargeting_time=should_bypass_retargeting_time,
    )

    # Get the consumed energy during the effective contact time
    consumed_energy = transmission_energy(
        power=OPTConfig.PEAK_TRANSMISSION_POWER,
        duration=effective_contact_time,
    )

    return consumed_energy
