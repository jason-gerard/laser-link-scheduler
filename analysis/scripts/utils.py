import math
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.constants import OPTConfig, MLConfig, DESTINATION_NODES
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


def compute_state_metrics_aggregated(
    teg: TimeExpandedGraph,
    state: int,
    accumulated_time: float,
    should_bypass_retargeting_time: bool,
) -> pd.DataFrame:
    state_duration = teg.state_durations[state]
    from_time = accumulated_time
    to_time = accumulated_time + state_duration

    oi_to_node_idx = np.array(
        [teg.optical_interfaces_to_node[i] for i in range(teg.N)],
        dtype=int,
    )
    oi_node_ids = np.array(
        [teg.nodes[node_idx].id for node_idx in oi_to_node_idx],
        dtype=object,
    )
    valid_tx_mask = ~np.isin(oi_node_ids, list(DESTINATION_NODES))
    valid_tx_indices = np.flatnonzero(valid_tx_mask)
    valid_tx_node_ids = oi_node_ids[valid_tx_indices]

    initial_powers = MLConfig.get_initial_powers(valid_tx_node_ids)
    generated_per_tx = generated_energy(
        from_time=from_time,
        to_time=to_time,
        initial_power=initial_powers,
        decay_constant=MLConfig.DECAY_RATE,
    )

    graph = np.asarray(teg.graphs[state])
    consumed_per_tx = np.zeros(len(valid_tx_indices), dtype=float)

    active_tx_local, rx_indices = np.where(graph[valid_tx_indices] == 1)

    for local_tx_pos, rx_oi_idx in zip(active_tx_local, rx_indices):
        tx_oi_idx = valid_tx_indices[local_tx_pos]
        consumed_per_tx[local_tx_pos] += transmission_energy(
            power=OPTConfig.PEAK_TRANSMISSION_POWER,
            duration=compute_effective_contact_time(
                oi_idx1=tx_oi_idx,
                oi_idx2=rx_oi_idx,
                scheduled_contact_topology=teg.graphs[:state],
                state_duration=state_duration,
                positions=teg.pos,
                optical_interfaces_to_node=teg.optical_interfaces_to_node,
                nodes=teg.nodes,
                should_bypass_retargeting_time=should_bypass_retargeting_time,
            ),
        )

    return pd.DataFrame(
        {
            "state_index": state,
            "tx_node_id": valid_tx_node_ids,
            "state_duration": state_duration,
            "generated_energy": generated_per_tx,
            "consumed_energy": consumed_per_tx,
        }
    )
