import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import typer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.scripts.utils import (  # noqa: E402
    AnalysisRunTable,
    load_report_tegs,
)
from src.constants import PLOTS_ROOT, RELAY_NODES  # noqa: E402
from src.models.pointing_delay import SLEW_RATE  # noqa: E402
from src.time_expanded_graph.time_expanded_graph import (  # noqa: E402
    TimeExpandedGraph,
)
from src.topology.weights import compute_all_delays  # noqa: E402


plt.rcParams.update({"font.size": 26})
plt.rc("legend", fontsize=22)
plt.rcParams.update({"font.family": "Times New Roman"})

app = typer.Typer()


def load_tegs(report_id: int) -> list[tuple[str, str, TimeExpandedGraph]]:
    return load_report_tegs(report_id)


def run_analysis(report_id: int) -> None:
    tegs = load_tegs(report_id)
    rows = [
        {
            "algorithm": algorithm,
            "scenario": scenario,
            "nodes": str(len(teg.node_map)),
            "states": str(teg.K),
            "samples": "-",
            "progress": "-",
        }
        for algorithm, scenario, teg in tegs
    ]
    columns = [
        ("algorithm", {"no_wrap": True}),
        ("scenario", {"no_wrap": True}),
        ("nodes", {"justify": "right", "no_wrap": True}),
        ("states", {"justify": "right", "no_wrap": True}),
        ("samples", {"justify": "right", "no_wrap": True}),
        ("progress", {"no_wrap": True}),
    ]

    with AnalysisRunTable(columns, rows) as run_table:
        for idx, (algorithm, scenario, teg) in enumerate(tegs):
            run_table.mark_progress(idx, 0)
            active_edges = np.argwhere(teg.graphs == 1)
            total_edges = len(active_edges)
            update_interval = max(1, total_edges // 100) if total_edges else 1
            sample_count = 0
            all_pointing_delays = []
            all_pointing_delays_with_node = []
            all_link_acq_delays = []

            for edge_idx, (k, tx_oi_idx, rx_oi_idx) in enumerate(active_edges):
                res = compute_all_delays(
                    int(tx_oi_idx),
                    int(rx_oi_idx),
                    teg.graphs[:k],
                    int(teg.state_durations[k]),
                    teg.pos,
                    teg.optical_interfaces_to_node,
                    teg.nodes,
                    SLEW_RATE,
                )
                if res is None:
                    continue

                link_acq_delay = res[0]
                pd1, idx1, idx2, idx1_rx = res[1]
                pd2, idx2, idx1, idx2_rx = res[2]
                idx1 = int(idx1)
                idx2 = int(idx2)
                idx1_rx = int(idx1_rx)
                idx2_rx = int(idx2_rx)
                pd1 += 2
                pd2 += 2

                idx1_node_id = teg.nodes[idx1].id
                idx2_node_id = teg.nodes[idx2].id
                idx1_rx_node_id = teg.nodes[idx1_rx].id
                idx2_rx_node_id = teg.nodes[idx2_rx].id

                all_pointing_delays.append(pd1)
                all_pointing_delays.append(pd2)
                all_link_acq_delays.append(link_acq_delay)
                sample_count += 2

                if not (
                    idx1_node_id in RELAY_NODES
                    and idx2_node_id in RELAY_NODES
                    and idx1_rx_node_id in RELAY_NODES
                ):
                    all_pointing_delays_with_node.append(
                        (
                            idx1_node_id,
                            idx2_node_id,
                            idx1_rx_node_id,
                            pd1,
                        )
                    )
                if not (
                    idx1_node_id in RELAY_NODES
                    and idx2_node_id in RELAY_NODES
                    and idx2_rx_node_id in RELAY_NODES
                ):
                    all_pointing_delays_with_node.append(
                        (
                            idx2_node_id,
                            idx1_node_id,
                            idx2_rx_node_id,
                            pd2,
                        )
                    )

                if (
                    edge_idx + 1
                ) % update_interval == 0 or edge_idx == total_edges - 1:
                    run_table.update(idx, "samples", str(sample_count))
                    run_table.mark_progress(
                        idx,
                        int(((edge_idx + 1) / max(total_edges, 1)) * 100),
                    )

            run_table.update(idx, "samples", str(sample_count))
            run_table.mark_progress(idx, 100)

            all_pointing_delays_np = np.array(all_pointing_delays)
            all_link_acq_delays_np = np.array(all_link_acq_delays)
            all_pointing_delays_with_node = sorted(
                all_pointing_delays_with_node, key=lambda x: x[-1]
            )

            all_pointing_delays_np = all_pointing_delays_np[
                all_pointing_delays_np > 0
            ]
            all_link_acq_delays_np = all_link_acq_delays_np[
                all_link_acq_delays_np > 0
            ]

            if (
                len(all_pointing_delays_np) == 0
                or len(all_link_acq_delays_np) == 0
            ):
                continue

            kde_pointing = gaussian_kde(all_pointing_delays_np)
            kde_acquisition = gaussian_kde(all_link_acq_delays_np)

            x_pointing = np.linspace(0, all_pointing_delays_np.max() + 5, 200)
            x_acquisition = np.linspace(
                0, all_link_acq_delays_np.max() + 20, 200
            )

            pointing_bins = np.linspace(
                2, all_pointing_delays_np.max() + 5, 40
            )
            acquisition_bins = np.linspace(
                2, all_link_acq_delays_np.max() + 20, 40
            )

            plt.figure(figsize=(10, 6))

            ax = plt.gca()
            ax.set_axisbelow(True)

            plt.hist(
                all_pointing_delays_np,
                bins=pointing_bins,
                density=True,
                ALPHA=0.4,
                color="blue",
                label="Pointing Delay Histogram",
            )
            plt.hist(
                all_link_acq_delays_np,
                bins=acquisition_bins,
                density=True,
                ALPHA=0.4,
                color="orange",
                label="Acquisition Delay Histogram",
            )

            plt.plot(
                x_pointing,
                kde_pointing(x_pointing),
                label="Pointing Delay PDF",
                color="blue",
            )
            plt.plot(
                x_acquisition,
                kde_acquisition(x_acquisition),
                label="Acquisition Delay PDF",
                color="orange",
            )

            plt.xlim(-0.25, 250)
            plt.xticks(np.arange(0, 251, 25))

            plt.ylim(0, 0.18)
            plt.yticks(np.arange(0, 0.17, 0.02))

            plt.xlabel("Delay (seconds)")
            plt.ylabel("Probability Density")
            plt.grid(linestyle="-", color="0.95")
            plt.legend()

            bbox = dict(boxstyle="round", fc="0.9")
            arrowprops = dict(
                arrowstyle="->",
                connectionstyle="angle,angleA=0,angleB=90,rad=10",
            )

            plt.annotate(
                "IPN-to-IPN",
                fontsize=20,
                xy=(3.0, 0.135),
                xytext=(4.0, 0.16),
                textcoords="data",
                bbox=bbox,
                arrowprops=arrowprops,
                ha="left",
                va="bottom",
            )
            plt.annotate(
                "Ground/LEO-to-IPN",
                fontsize=20,
                xy=(37, 0.02),
                xytext=(50, 0.13),
                textcoords="data",
                bbox=bbox,
                arrowprops=arrowprops,
                ha="center",
                va="bottom",
            )
            plt.annotate(
                "LEO-to-LEO",
                fontsize=20,
                xy=(85, 0.015),
                xytext=(85, 0.05),
                textcoords="data",
                bbox=bbox,
                arrowprops=arrowprops,
                ha="center",
                va="bottom",
            )
            plt.annotate(
                "LEO Acq",
                fontsize=20,
                xy=(48.0, 0.085),
                xytext=(46.0, 0.1),
                textcoords="data",
                bbox=bbox,
                arrowprops=arrowprops,
                ha="left",
                va="bottom",
            )
            plt.annotate(
                "IPN Acq",
                fontsize=20,
                xy=(210.0, 0.012),
                xytext=(210.0, 0.04),
                textcoords="data",
                bbox=bbox,
                arrowprops=arrowprops,
                ha="center",
                va="bottom",
            )

            plt.tight_layout()

            plot_dir = os.path.join(PLOTS_ROOT, scenario, algorithm)
            os.makedirs(plot_dir, exist_ok=True)
            file_name = "retargeting_delay_pdf".replace(" ", "_").replace(
                "/", "_"
            )
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
            plt.close()


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
