import math
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx
import numpy as np
import typer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.time_expanded_graph.time_expanded_graph import (  # noqa: E402
    TimeExpandedGraph,
)
from src.topology.contact_plan import IONContactPlanParser  # noqa: E402
from analysis.scripts.utils import AnalysisRunTable  # noqa: E402


SHOW_FIGS = False

EXPERIMENT_NAME = "mars_earth_xs_scenario"
DEFAULT_EXPERIMENT_NAMES = [
    "gs_mars_earth_scenario_inc_4",
    "gs_mars_earth_scenario_inc_8",
    "gs_mars_earth_scenario_inc_12",
    "gs_mars_earth_scenario_inc_16",
    "gs_mars_earth_scenario_inc_20",
    "gs_mars_earth_scenario_inc_24",
    "gs_mars_earth_scenario_inc_28",
    "gs_mars_earth_scenario_inc_32",
    "gs_mars_earth_scenario_inc_36",
    "gs_mars_earth_scenario_inc_40",
    "gs_mars_earth_scenario_inc_44",
    "gs_mars_earth_scenario_inc_48",
    "gs_mars_earth_scenario_inc_52",
    "gs_mars_earth_scenario_inc_56",
    "gs_mars_earth_scenario_inc_60",
    "gs_mars_earth_scenario_inc_64",
]

app = typer.Typer()

plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})


def count_reduced_edges(experiment_name):
    contact_plan_parser = IONContactPlanParser()
    contact_plan = contact_plan_parser.read(experiment_name)

    teg = TimeExpandedGraph.from_contact_plan(
        contact_plan, should_fractionate=False
    )
    teg_count = teg.count_edges()

    frac_teg = teg.fractionate_graph()
    frac_teg_count = frac_teg.count_edges()

    reduced_teg = frac_teg.dag_reduction()
    reduced_teg_count = reduced_teg.count_edges()

    print(teg_count, frac_teg_count, reduced_teg_count)
    print(
        f"Percent of edges removed = {100 * (1 - reduced_teg.count_edges() / frac_teg.count_edges()):.3f}%"
    )

    return teg_count, frac_teg_count, reduced_teg_count
    # if SHOW_FIGS:
    #     visualize(teg, name="teg")
    #     visualize(reduced_teg, name="reduced_teg")
    # plt.show()


def visualize(teg, name):
    rand = np.random.randint(1, 10)
    for k in range(1):
        num_nodes = teg.N

        edges = []
        for tx_idx in range(num_nodes):
            for rx_idx in range(num_nodes):
                if teg.graphs[k][tx_idx][rx_idx] >= 1:
                    edges.append((teg.nodes[tx_idx], teg.nodes[rx_idx]))

        G = nx.DiGraph()
        for node in teg.nodes:
            G.add_node(node)
        G.add_edges_from(edges)

        # print(teg.graphs[k])
        plt.figure(k + rand)
        # nx.draw(G, nx.spring_layout(G), node_size=1500, with_labels=False)
        A = nx.nx_agraph.to_agraph(G)
        A.layout(prog="dot")
        A.draw(f"{name}.png", args="-Gnodesep=0.01 -Gfont_size=1", prog="dot")


@app.command()
def main(
    experiment_names: list[str] = typer.Option(
        DEFAULT_EXPERIMENT_NAMES,
        "--experiment-name",
        "-e",
        help="Experiment name to analyze. Pass multiple times for multiple experiments.",
    ),
) -> None:
    x = [int(name.split("_")[-1]) for name in experiment_names]

    teg_counts = []
    frac_teg_counts = []
    reduced_teg_counts = []
    rows = [
        {
            "experiment": name,
            "teg": "-",
            "fractionated": "-",
            "reduced": "-",
            "progress": "-",
        }
        for name in experiment_names
    ]
    columns = [
        ("experiment", {"no_wrap": True}),
        ("teg", {"justify": "right", "no_wrap": True}),
        ("fractionated", {"justify": "right", "no_wrap": True}),
        ("reduced", {"justify": "right", "no_wrap": True}),
        ("progress", {"no_wrap": True}),
    ]

    with AnalysisRunTable(columns, rows) as run_table:
        for idx, name in enumerate(experiment_names):
            run_table.mark_progress(idx, 0)
            teg_count, frac_teg_count, reduced_teg_count = count_reduced_edges(
                name
            )
            teg_counts.append(teg_count)
            frac_teg_counts.append(frac_teg_count)
            reduced_teg_counts.append(reduced_teg_count)
            run_table.update(idx, "teg", str(teg_count))
            run_table.update(idx, "fractionated", str(frac_teg_count))
            run_table.update(idx, "reduced", str(reduced_teg_count))
            run_table.mark_progress(idx, 100)

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot()

    ax1.set_xticks([i for i in x if i % 8 == 0])
    ax1.set_xticklabels([f"{i}/{math.ceil(i / 16)}" for i in x if i % 8 == 0])

    ax1.plot(x, teg_counts, label="Standard TEG", linewidth=2.5)
    ax1.plot(x, frac_teg_counts, label="Fractionated TEG", linewidth=2.5)
    ax1.plot(x, reduced_teg_counts, label="Reduced TEG", linewidth=2.5)

    label = "Number of decision variables"
    ax1.set_ylabel(label)
    ax1.set_xlabel("Source/relay node counts")

    ax1.set_yscale("log")

    plt.grid(linestyle="-", color="0.95")

    ax2 = ax1.twinx()

    y2 = [
        float(f"{100 * (1 - reduced_teg_counts[i] / frac_teg_counts[i]): .3f}")
        for i in range(len(teg_counts))
    ]
    ax2.plot(
        x,
        y2,
        "tab:cyan",
        linestyle="dashed",
        label="% Edges Removed",
        linewidth=2.5,
    )
    ax2.set_ylabel("Percent reduction")
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    # Get the handles and labels from both axes
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine the handles and labels
    all_handles = handles + handles2
    all_labels = labels + labels2

    # Create a single legend
    plt.legend(all_handles, all_labels, loc="lower right")

    file_name = label.replace(" ", "_").replace("/", "_")
    plt.savefig(
        os.path.join("analysis", f"{file_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join("analysis", f"{file_name}.png"),
        format="png",
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    app()
