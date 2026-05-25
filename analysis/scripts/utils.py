import math
import os
from pathlib import Path
import pickle
import sys
from collections.abc import Callable
from typing import Any
import numpy as np
import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.constants import (
    OCTConfig,
    MLConfig,
    DESTINATION_NODES,
    REPORTS_ROOT,
)
from src.models import (
    mission_lifetime,
    survivability,
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
        enable_live: bool = True,
    ) -> None:
        self.columns = columns
        self.rows = rows
        self.refresh_per_second = refresh_per_second
        self.enable_live = enable_live
        self.console = Console()
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
        if self.enable_live:
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
        else:
            self.console.print(self.build_table())

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
    baseline_energy = (
        MLConfig.get_baseline_powers(valid_tx_node_ids) * state_duration
    )

    active_tx, active_rx = np.where(graph[valid_tx_indices] == 1)
    for local_tx_pos, rx_oi_idx in zip(active_tx, active_rx):
        tx_oi_idx = valid_tx_indices[local_tx_pos]
        transmission_energy_consumed = transmission_energy(
            power=OCTConfig.PEAK_TRANSMISSION_POWER,
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
        consumed_per_tx[local_tx_pos] += (
            transmission_energy_consumed + baseline_energy[local_tx_pos]
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


def estimate_lifetime_from_average_load(
    node_id: str,
    average_power_load: float,
) -> float:
    if node_id in DESTINATION_NODES:
        return float("inf")
    if average_power_load <= 0:
        return float("inf")

    initial_power = MLConfig.get_initial_power(node_id)
    baseline_power = MLConfig.get_baseline_power(node_id)
    return float(
        mission_lifetime(
            P0=initial_power,
            decay_constant=MLConfig.DECAY_RATE,
            P_min=average_power_load + baseline_power,
        )
    )


def compute_lifetime_metrics(
    teg: TimeExpandedGraph,
    should_bypass_retargeting_time: bool,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    mission_duration = float(np.sum(teg.state_durations))
    accumulated_time = 0.0
    state_metrics: list[pd.DataFrame] = []

    for state in range(teg.K):
        state_df = compute_state_metrics_aggregated(
            teg=teg,
            state=state,
            accumulated_time=accumulated_time,
            should_bypass_retargeting_time=should_bypass_retargeting_time,
        )
        aggregated_state_df = state_df.groupby(
            "tx_node_id", as_index=False
        ).agg(
            generated_energy=("generated_energy", "first"),
            consumed_energy=("consumed_energy", "sum"),
        )
        aggregated_state_df["state_index"] = state
        state_metrics.append(aggregated_state_df)
        accumulated_time += float(teg.state_durations[state])

        if progress_callback is not None:
            progress_callback(state + 1, teg.K)

    if not state_metrics:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "node_id",
                    "initial_power",
                    "mission_duration",
                    "total_generated_energy",
                    "total_consumed_energy",
                    "net_energy",
                    "average_power_load",
                    "final_generated_power",
                    "estimated_lifetime",
                    "estimated_lifetime_years",
                    "depleted_within_horizon",
                ]
            )
        )

    total_df = pd.concat(state_metrics, ignore_index=True)
    total_df = total_df.groupby("tx_node_id", as_index=False).agg(
        total_generated_energy=("generated_energy", "sum"),
        total_consumed_energy=("consumed_energy", "sum"),
    )
    total_df = total_df.rename(columns={"tx_node_id": "node_id"})
    total_df["initial_power"] = total_df["node_id"].map(
        MLConfig.get_initial_power
    )
    total_df["mission_duration"] = mission_duration
    total_df["net_energy"] = (
        total_df["total_generated_energy"] - total_df["total_consumed_energy"]
    )
    total_df["average_power_load"] = np.where(
        mission_duration > 0,
        total_df["total_consumed_energy"] / mission_duration,
        0.0,
    )
    total_df["final_generated_power"] = total_df["initial_power"] * np.exp(
        -MLConfig.DECAY_RATE * mission_duration
    )
    total_df["estimated_lifetime"] = total_df.apply(
        lambda row: estimate_lifetime_from_average_load(
            str(row["node_id"]),
            float(row["average_power_load"]),
        ),
        axis=1,
    )

    total_df["estimated_lifetime_years"] = total_df["estimated_lifetime"] / (
        365.25 * 24 * 60 * 60
    )
    total_df["depleted_within_horizon"] = np.isfinite(
        total_df["estimated_lifetime"]
    ) & (total_df["estimated_lifetime"] <= mission_duration)
    total_df.loc[total_df["depleted_within_horizon"], "estimated_lifetime"] = (
        0.0
    )
    total_df.loc[
        total_df["depleted_within_horizon"], "estimated_lifetime_years"
    ] = 0.0

    return total_df.sort_values("node_id").reset_index(drop=True)


def summarize_lifetime_metrics(
    metrics_df: pd.DataFrame,
) -> dict[str, float | int]:
    finite_lifetimes = metrics_df.loc[
        np.isfinite(metrics_df["estimated_lifetime"]), "estimated_lifetime"
    ]
    finite_lifetimes_years = metrics_df.loc[
        np.isfinite(metrics_df["estimated_lifetime_years"]),
        "estimated_lifetime_years",
    ]
    mission_duration = (
        float(metrics_df["mission_duration"].iloc[0])
        if not metrics_df.empty
        else 0.0
    )

    return {
        "spacecraft_count": int(len(metrics_df)),
        "mission_duration": mission_duration,
        "total_generated_energy": float(
            metrics_df["total_generated_energy"].sum()
        ),
        "total_consumed_energy": float(
            metrics_df["total_consumed_energy"].sum()
        ),
        "min_estimated_lifetime": float(
            finite_lifetimes.min() if not finite_lifetimes.empty else np.inf
        ),
        "min_estimated_lifetime_years": float(
            finite_lifetimes_years.min()
            if not finite_lifetimes_years.empty
            else np.inf
        ),
        "mean_estimated_lifetime": float(
            finite_lifetimes.mean() if not finite_lifetimes.empty else np.inf
        ),
        "mean_estimated_lifetime_years": float(
            finite_lifetimes_years.mean()
            if not finite_lifetimes_years.empty
            else np.inf
        ),
        "median_estimated_lifetime": float(
            finite_lifetimes.median() if not finite_lifetimes.empty else np.inf
        ),
        "median_estimated_lifetime_years": float(
            finite_lifetimes_years.median()
            if not finite_lifetimes_years.empty
            else np.inf
        ),
        "depleted_spacecraft_within_horizon": int(
            metrics_df["depleted_within_horizon"].sum()
        ),
        "non_depleting_spacecraft": int(
            (~np.isfinite(metrics_df["estimated_lifetime"])).sum()
        ),
    }
