import math
import os
from pathlib import Path
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import typer
from rich import print
from rich.live import Live
from rich.table import Table
import pandas as pd

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

app = typer.Typer()


def load_tegs(report_id: int) -> list[tuple[str, str, TimeExpandedGraph]]:
    tegs = []
    report_dir = os.path.join(REPORTS_ROOT, str(report_id))

    for file_name in os.listdir(report_dir):
        if not file_name.endswith(".pkl"):
            continue

        file_path = os.path.join(report_dir, file_name)
        match = file_name.split(".")[0].split("_")
        algorithm = match[0]
        scenario = "_".join(match[1:])
        if algorithm not in [alg[0] for alg in algorithms]:
            continue

        with open(file_path, "rb") as f:
            teg: TimeExpandedGraph = pickle.load(f)
        tegs.append((algorithm, scenario, teg))

    return tegs


def run_analysis(report_id: int) -> None:
    tegs = load_tegs(report_id)
    rows: list[dict[str, str]] = [
        {
            "algorithm": algorithm,
            "scenario": scenario,
            "nodes": str(len(teg.node_map)),
            "states": str(teg.K),
            "progress": "-",
        }
        for algorithm, scenario, teg in tegs
    ]

    def build_table() -> Table:
        table = Table(expand=False)
        table.add_column("algorithm", no_wrap=True)
        table.add_column("scenario", no_wrap=True)
        table.add_column("nodes", justify="right", no_wrap=True)
        table.add_column("states", justify="right", no_wrap=True)
        table.add_column("progress", no_wrap=True)
        for row in rows:
            table.add_row(
                row["algorithm"],
                row["scenario"],
                row["nodes"],
                row["states"],
                row["progress"],
            )
        return table

    with Live(build_table(), refresh_per_second=8, transient=False) as live:
        for idx, (algorithm, scenario, teg) in enumerate(tegs):
            rows[idx]["progress"] = "0%"
            live.update(build_table(), refresh=True)

            EFFECTIVE_CONTACT_TIME_CACHE.clear()
            COORDINATE_CACHE.clear()

            should_bypass_retargeting_time = algorithm == "lls-pat-unaware"
            state_metrics_list = []
            accumulated_time = 0.0
            update_interval = max(1, teg.K // 100)

            for state in range(teg.K):
                state_df = compute_state_metrics_aggregated(
                    teg=teg,
                    state=state,
                    accumulated_time=accumulated_time,
                    should_bypass_retargeting_time=should_bypass_retargeting_time,
                )
                state_metrics_list.append(state_df)
                accumulated_time += teg.state_durations[state]

                if (state + 1) % update_interval == 0 or state == teg.K - 1:
                    rows[idx]["progress"] = (
                        f"{int(((state + 1) / teg.K) * 100)}%"
                    )
                    live.update(build_table(), refresh=True)

            total_metrics_df = [
                state_df.groupby("tx_node_id", as_index=False).agg(
                    state_index=("state_index", "first"),
                    state_duration=("state_duration", "first"),
                    generated_energy=("generated_energy", "first"),
                    consumed_energy=("consumed_energy", "sum"),
                )
                for state_df in state_metrics_list
            ]

            metrics_rows = []
            for df in total_metrics_df:
                total_generated = df["generated_energy"].sum()
                total_consumed = df["consumed_energy"].sum()

                metrics_rows.append(
                    {
                        "total_time": df["state_duration"].max(),
                        "total_generated": total_generated,
                        "total_consumed": total_consumed,
                        "total_additional_energy": total_generated
                        - total_consumed,
                    }
                )

            total_state_metrics = pd.DataFrame(metrics_rows)
            total_state_metrics.index.name = "state_index"

            total_state_metrics.to_csv(
                f"{REPORTS_ROOT}/{report_id}/energy_metrics_{algorithm}_{scenario}.csv"
            )

            plot_dir = os.path.join(PLOTS_ROOT, str(f"{algorithm}_{scenario}"))
            os.makedirs(plot_dir, exist_ok=True)

            total_state_metrics.plot(
                y=[
                    "total_generated",
                    "total_consumed",
                    "total_additional_energy",
                ],
                kind="line",
                title=f"Energy Metrics for {algorithm} with {scenario} nodes",
            )
            plt.xlabel("State Index")
            plt.ylabel("Energy (J)")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                f"{PLOTS_ROOT}/{algorithm}_{scenario}/energy_metrics_{algorithm}_{scenario}.png"
            )
            plt.close()

            rows[idx]["progress"] = "100%"
            live.update(build_table(), refresh=True)
        # If we want to localize the results in the report instance.
        # plt.savefig(
        #     f"{PLOTS_ROOT}/{report_id}/energy_metrics_{algorithm}_{scenario}.png"
        # )


@app.command()
def main(
    report_id: int = typer.Option(
        ...,
        "--report-id",
        "-r",
        help="Report identifier used to read inputs from reports/ and write plots/",
    ),
) -> None:
    run_analysis(report_id)


if __name__ == "__main__":
    app()
