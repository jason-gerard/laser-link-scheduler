import math
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import typer
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.scripts.utils import (
    AnalysisRunTable,
    compute_state_metrics_aggregated,
    load_report_tegs,
    normalize_algorithm_name,
)
from src.time_expanded_graph.time_expanded_graph import (
    TimeExpandedGraph,
)
from src.constants import DESTINATION_NODES, PLOTS_ROOT
from src.topology.weights import EFFECTIVE_CONTACT_TIME_CACHE, COORDINATE_CACHE

plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

app = typer.Typer()


def load_tegs(report_id: int) -> list[tuple[str, str, TimeExpandedGraph]]:
    return sorted(load_report_tegs(report_id))


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

    columns = [
        ("algorithm", {"no_wrap": True}),
        ("scenario", {"no_wrap": True}),
        ("nodes", {"justify": "right", "no_wrap": True}),
        ("states", {"justify": "right", "no_wrap": True}),
        ("progress", {"no_wrap": True}),
    ]

    with AnalysisRunTable(columns, rows) as run_table:
        for idx, (algorithm, scenario, teg) in enumerate(tegs):
            run_table.mark_progress(idx, 0)

            should_bypass_retargeting_time = (
                normalize_algorithm_name(algorithm) == "lls_pat_unaware"
            )
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
                    run_table.mark_progress(
                        idx,
                        int(((state + 1) / teg.K) * 100),
                    )

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

            plot_dir = os.path.join(PLOTS_ROOT, scenario, algorithm)
            os.makedirs(plot_dir, exist_ok=True)
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(10, 7),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )

            x = total_state_metrics.index.to_numpy()
            ax1.plot(
                x,
                total_state_metrics["total_generated"],
                label="Total generated",
                linewidth=2,
            )
            ax1.plot(
                x,
                total_state_metrics["total_consumed"],
                label="Total consumed",
                linewidth=2,
            )
            ax1.plot(
                x,
                total_state_metrics["total_additional_energy"],
                label="Total additional energy",
                linewidth=2,
            )
            ax1.set_xlabel("State Index")
            ax1.set_ylabel("Energy (J)")
            ax1.set_title(
                f"State Energy Analysis for {algorithm} on {scenario}"
            )
            ax1.grid()
            ax2.plot(
                x,
                total_state_metrics["total_time"],
                color="tab:purple",
                linestyle="-",
                linewidth=2,
                label="State duration",
            )
            ax2.set_ylabel("Duration [s]")
            ax2.set_xlabel("State Index")
            ax2.grid()

            ax1.legend(loc="upper right")
            ax2.legend(loc="upper right")

            plt.tight_layout()
            plt.savefig(
                f"{PLOTS_ROOT}/{scenario}/{algorithm}/energy_metrics.png"
            )
            plt.close(fig)

            run_table.mark_progress(idx, 100)
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
        help="Report identifier used to read scheduled TEGs and write state energy plots.",
    ),
) -> None:
    run_analysis(report_id)


if __name__ == "__main__":
    app()
