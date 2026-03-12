import math
import os
from pathlib import Path
import pickle
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.constants import OPTConfig, MLConfig
from src.models import mission_lifetime, generating_power


def compute_energy_metrics(
    teg,
    tx_oi_idx,
    rx_oi_idx,
    state_contact_topology,
    state_duration,
    acumulated_time,
    tx_node_id,
):
    from_time = acumulated_time
    to_time = acumulated_time + state_duration

    initial_power = MLConfig.get_initial_power(tx_node_id)

    generated: float = _get_generated_energy(
        state_duration, from_time, to_time, initial_power
    )
    consumed: float = _get_consumed_energy(from_time, to_time)

    return generated, consumed


def _get_generated_energy(state_duration, from_time, to_time, initial_power):
    state_initial_power = generating_power(
        from_time, initial_power, MLConfig.DECAY_RATE
    )
    state_final_power = generating_power(
        from_time, initial_power, MLConfig.DECAY_RATE
    )
    delta_power = state_initial_power - state_final_power
    return delta_power * state_duration


def _get_consumed_energy(from_time, to_time):
    return 0.0
