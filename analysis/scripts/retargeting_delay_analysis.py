import math
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import typer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.scripts.utils import (  # noqa: E402
    AnalysisRunTable,
    load_report_tegs,
)
from src.constants import PLOTS_ROOT  # noqa: E402
from src.time_expanded_graph.time_expanded_graph import (  # noqa: E402
    TimeExpandedGraph,
)
from src.topology.weights import compute_delays  # noqa: E402


plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

ALGORITHMS = ["lls", "lls_pat_unaware", "lls_mip", "fcp"]

app = typer.Typer()


def load_tegs(report_id: int) -> list[tuple[str, str, TimeExpandedGraph]]:
    return load_report_tegs(report_id, allowed_algorithms=ALGORITHMS)


def run_analysis(report_id: int) -> None:
    tegs = load_tegs(report_id)
    rows = [
        {
            "algorithm": algorithm,
            "scenario": scenario,
            "nodes": str(len(teg.node_map)),
            "states": str(teg.K),
            "duty_cycle": "-",
            "progress": "-",
        }
        for algorithm, scenario, teg in tegs
    ]
    columns = [
        ("algorithm", {"no_wrap": True}),
        ("scenario", {"no_wrap": True}),
        ("nodes", {"justify": "right", "no_wrap": True}),
        ("states", {"justify": "right", "no_wrap": True}),
        ("duty_cycle", {"justify": "right", "no_wrap": True}),
        ("progress", {"no_wrap": True}),
    ]

    with AnalysisRunTable(columns, rows) as run_table:
        for idx, (algorithm, scenario, teg) in enumerate(tegs):
            run_table.mark_progress(idx, 0)
            delays_by_node = {node: [] for node in teg.nodes}
            active_edges = np.argwhere(teg.graphs == 1)
            total_edges = len(active_edges)
            update_interval = max(1, total_edges // 100) if total_edges else 1

            for edge_idx, (k, tx_oi_idx, rx_oi_idx) in enumerate(active_edges):
                pointing_delay, link_acq_delay = compute_delays(
                    int(tx_oi_idx),
                    int(rx_oi_idx),
                    teg.graphs[:k],
                    int(teg.state_durations[k]),
                    teg.pos,
                    teg.optical_interfaces_to_node,
                    teg.nodes,
                )

                tx_node = teg.nodes[
                    teg.optical_interfaces_to_node[int(tx_oi_idx)]
                ]
                rx_node = teg.nodes[
                    teg.optical_interfaces_to_node[int(rx_oi_idx)]
                ]
                edge_data = (
                    int(teg.state_durations[k]),
                    pointing_delay,
                    link_acq_delay,
                )
                delays_by_node[tx_node].append(edge_data)
                delays_by_node[rx_node].append(edge_data)

                if (
                    edge_idx + 1
                ) % update_interval == 0 or edge_idx == total_edges - 1:
                    run_table.mark_progress(
                        idx,
                        int(((edge_idx + 1) / max(total_edges, 1)) * 100),
                    )

            network_total_time = 0.0
            network_total_eff_time = 0.0
            for node_delays in delays_by_node.values():
                total_time = 0.0
                total_eff_time = 0.0
                for (
                    state_duration,
                    pointing_delay,
                    link_acq_delay,
                ) in node_delays:
                    total_time += state_duration
                    total_eff_time += state_duration - (
                        pointing_delay + link_acq_delay
                    )
                    network_total_time += state_duration
                    network_total_eff_time += state_duration - (
                        pointing_delay + link_acq_delay
                    )

                if total_time == 0:
                    continue

            network_retargeting_duty_cycle = (
                network_total_eff_time / network_total_time
                if network_total_time
                else 0.0
            )
            run_table.update(
                idx,
                "duty_cycle",
                f"{network_retargeting_duty_cycle:.4f}",
            )
            run_table.mark_progress(idx, 100)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.bar(
                [algorithm],
                [network_retargeting_duty_cycle],
                width=0.6,
            )
            plt.ylabel("Retargeting Duty Cycle [%]")
            plt.ylim(0.0, 1.0)
            plt.grid(linestyle="-", color="0.95", axis="y")
            plt.tight_layout()

            plot_dir = os.path.join(PLOTS_ROOT, scenario, algorithm)
            os.makedirs(plot_dir, exist_ok=True)
            file_name = "network_retargeting_duty_cycle"
            plt.savefig(
                os.path.join(plot_dir, f"{file_name}.pdf"),
                format="pdf",
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(plot_dir, f"{file_name}.png"),
                format="png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig)


@app.command()
def main(
    report_id: int = typer.Option(
        ...,
        "--report-id",
        "-r",
        help="Report identifier used to read TEGs from reports/.",
    ),
) -> None:
    run_analysis(report_id)


if __name__ == "__main__":
    app()
