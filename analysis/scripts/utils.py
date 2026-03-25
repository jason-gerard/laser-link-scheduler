import math
import os
from pathlib import Path
import pickle
import sys
from typing import Any
import numpy as np
import pandas as pd
from rich.live import Live
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.constants import (
    OPTConfig,
    MLConfig,
    DESTINATION_NODES,
    REPORTS_ROOT,
)
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


class AnalysisRunTable:
    def __init__(
        self,
        columns: list[tuple[str, dict[str, Any] | None]],
        rows: list[dict[str, str]],
        refresh_per_second: int = 8,
    ) -> None:
        self.columns = columns
        self.rows = rows
        self.refresh_per_second = refresh_per_second
        self.live: Live | None = None

    def build_table(self) -> Table:
        table = Table(expand=False)
        for name, options in self.columns:
            table.add_column(name, **(options or {}))
        for row in self.rows:
            table.add_row(
                *[row.get(column_name, "") for column_name, _ in self.columns]
            )
        return table

    def __enter__(self) -> "AnalysisRunTable":
        self.live = Live(
            self.build_table(),
            refresh_per_second=self.refresh_per_second,
            transient=False,
        )
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.live is not None:
            self.live.__exit__(exc_type, exc, tb)
            self.live = None

    def update(self, row_idx: int, key: str, value: str) -> None:
        self.rows[row_idx][key] = value
        if self.live is not None:
            self.live.update(self.build_table(), refresh=True)

    def mark_progress(self, row_idx: int, progress: int) -> None:
        self.update(row_idx, "progress", f"{progress}%")


def normalize_algorithm_name(name: str) -> str:
    return name.replace("-", "_")


def load_report_tegs(
    report_id: int,
    allowed_algorithms: list[str] | None = None,
) -> list[tuple[str, str, TimeExpandedGraph]]:
    tegs = []
    report_dir = os.path.join(REPORTS_ROOT, str(report_id))
    normalized_allowed_algorithms = (
        {normalize_algorithm_name(name) for name in allowed_algorithms}
        if allowed_algorithms is not None
        else None
    )

    for file_name in os.listdir(report_dir):
        if not file_name.endswith(".pkl"):
            continue

        stem = file_name.split(".")[0]
        parts = stem.split("_")
        algorithm = parts[0]
        scenario = "_".join(parts[1:])

        if (
            normalized_allowed_algorithms is not None
            and normalize_algorithm_name(algorithm)
            not in normalized_allowed_algorithms
        ):
            continue

        with open(os.path.join(report_dir, file_name), "rb") as f:
            teg: TimeExpandedGraph = pickle.load(f)
        tegs.append((algorithm, scenario, teg))

    return tegs


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
