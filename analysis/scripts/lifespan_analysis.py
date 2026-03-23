import math
import os
from pathlib import Path
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from rich import print
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.scripts.utils import compute_state_metrics_aggregated
from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
)
from src.constants import DESTINATION_NODES, REPORTS_ROOT, PLOTS_ROOT
from src.topology.weights import EFFECTIVE_CONTACT_TIME_CACHE, COORDINATE_CACHE

plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

algorithms = [
    ("lls", "LLS_Greedy"),
    ("lls-pat-unaware", "LLS_Greedy (ZRK)"),
    ("lls-mip", "LLS_MIP"),
    ("fcp", "FCP"),
]

report_id = 1773495127
tegs = []
report_dir = os.path.join(REPORTS_ROOT, str(report_id))
plot_dir = os.path.join(PLOTS_ROOT, str(report_id))
os.makedirs(plot_dir, exist_ok=True)
for file_name in os.listdir(report_dir):
    if file_name.endswith(".pkl"):
        file_path = os.path.join(report_dir, file_name)
        match = file_name.split(".")[0].split("_")
        algorithm = match[0]
        scenario = "_".join(match[1:])
        if algorithm not in [alg[0] for alg in algorithms]:
            continue
        with open(file_path, "rb") as f:
            teg: TimeExpandedGraph = pickle.load(f)
        tegs.append((algorithm, scenario, teg))

all_generation = []
all_consumption = []
for algorithm, scenario, teg in tegs:
    EFFECTIVE_CONTACT_TIME_CACHE.clear()
    COORDINATE_CACHE.clear()

    should_bypass_retargeting_time = algorithm == "lls-pat-unaware"
    print(
        f"Processing {algorithm} scenario {scenario} with {teg.N} nodes and {teg.K} states..."
    )
    # Initialize metrics storage
    state_metrics_list = []
    accumulated_time = 0.0

    for state in tqdm(range(teg.K)):
        state_df = compute_state_metrics_aggregated(
            teg=teg,
            state=state,
            accumulated_time=accumulated_time,
            should_bypass_retargeting_time=should_bypass_retargeting_time,
        )
        state_metrics_list.append(state_df)
        accumulated_time += teg.state_durations[state]

    total_metrics_df = state_metrics_list
    total_metrics_df = [
        state_df.groupby("tx_node_id", as_index=False).agg(
            state_index=("state_index", "first"),
            state_duration=("state_duration", "first"),
            generated_energy=("generated_energy", "first"),
            consumed_energy=("consumed_energy", "sum"),
        )
        for state_df in state_metrics_list
    ]

    rows = []
    for df in total_metrics_df:
        total_generated = df["generated_energy"].sum()
        total_consumed = df["consumed_energy"].sum()

        rows.append(
            {
                "total_time": df["state_duration"].max(),
                "total_generated": total_generated,
                "total_consumed": total_consumed,
                "total_additional_energy": total_generated - total_consumed,
            }
        )

    total_state_metrics = pd.DataFrame(rows)
    total_state_metrics.index.name = "state_index"

    # CSV
    total_state_metrics.to_csv(
        f"{REPORTS_ROOT}/{report_id}/energy_metrics_{algorithm}_{scenario}.csv"
    )

    # PLOT
    total_state_metrics.plot(
        y=["total_generated", "total_consumed", "total_additional_energy"],
        kind="line",
        title=f"Energy Metrics for {algorithm} with {scenario} nodes",
    )
    plt.xlabel("State Index")
    plt.ylabel("Energy (J)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{PLOTS_ROOT}/{report_id}/energy_metrics_{algorithm}_{scenario}.png"
    )
    plt.close()
    # If we want to globalize the results across all algorithms and node counts.
    # plt.savefig(f"{PLOTS_ROOT}/{algorithm}_{node_count}/energy_metrics_{algorithm}_{node_count}.png")
