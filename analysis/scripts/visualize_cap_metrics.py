import csv
import glob
import math
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import (
    DESTINATION_NODES,
    PLOTS_ROOT,
    RELAY_NODES,
    REPORTS_ROOT,
    SOURCE_NODES,
)

plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

# Unit conversion: internal capacity unit → terabits
# Internal bit-rate unit = 267,000 bps (267 kbps).
# capacity [internal] * 267_000 / 1e12 = capacity [terabits]
_BITS_PER_INTERNAL_UNIT = 267_000
_BITS_PER_TERABIT = 1_000_000_000_000


def to_terabits(raw: float) -> float:
    return raw * _BITS_PER_INTERNAL_UNIT / _BITS_PER_TERABIT


if len(sys.argv) < 2:
    print("Usage: python visualize_metrics_incremental_runs.py <report_id>")
    sys.exit(1)

report_id = int(sys.argv[1])
path = os.path.join(REPORTS_ROOT, str(report_id), f"{report_id}_report.csv")
with open(path, "r") as f:
    report = [
        {k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)
    ]

scenarios = []
for run in report:
    num_nodes = int(run["Scenario"].split("_")[-1])
    scenarios.append(num_nodes)

    # "Capacity by node" must be computed before overwriting "Capacity"
    run["Capacity by node"] = to_terabits(float(run["Capacity"])) / num_nodes
    run["Capacity"] = to_terabits(float(run["Capacity"]))
    run["Wasted capacity"] = to_terabits(float(run["Wasted capacity"]))
    run["Wasted buffer capacity"] = to_terabits(
        float(run["Wasted buffer capacity"])
    )
    run["Scheduled delay"] = float(run["Scheduled delay"]) / 3600  # s → hours
    run["Jain's fairness index"] = float(run["Jain's fairness index"])
    run["Execution duration"] = float(run["Execution duration"])

x = sorted(list(set(scenarios)))

# ---------------------------------------------------------------------------
# Algorithm display config — mirrors visualize_metrics_incremental_runs.py
# (algorithm_key, display_name, linestyle, linewidth, color)
# ---------------------------------------------------------------------------
algorithms = [
    # Baselines
    ("lls", "LLS_Greedy", "solid", 2.5, None),
    ("lls_pat_unaware", "LLS_Greedy (ZRK)", "dashed", 2.5, None),
    ("fcp", "FCP", "solid", 2.5, None),
    # Lifespan-aware algorithms
    ("energy_aware", "Energy-Aware", "solid", 2.5, None),
    ("battery_energy", "Battery-Aware", "dashed", 2.5, None),
    ("lifespan_aware", "Lifespan-Aware", "dotted", 2.5, None),
]

# (metric_key, unit, y_min, y_max, y_step)
metrics = [
    ("Capacity", "terabits/day", 0, None, 5),
    ("Capacity by node", "terabits/day", 0, None, 0.5),
    ("Wasted capacity", "terabits/day", 0, None, 10),
    ("Wasted buffer capacity", "terabits/day", 0, None, 5),
    ("Scheduled delay", "hours", 0, None, 1),
    ("Jain's fairness index", "", 0, 1.0, 0.1),
    ("Execution duration", "seconds", 0, 1e2, 30),
]

# DTE capacity baseline: count GS nodes from the first scheduled TEG in the
# report, so the line stays correct regardless of how many ground stations
# the scenario actually uses.
_dest_ids = set(DESTINATION_NODES)
_source_ids = set(SOURCE_NODES)
_relay_ids = set(RELAY_NODES)
_pkl_files = sorted(
    glob.glob(os.path.join(REPORTS_ROOT, str(report_id), "*.pkl"))
)

# Build "source/relay" x-axis labels from actual TEG node counts.
# Load one PKL per unique num_nodes to get the real counts.
node_count_to_label: dict[int, str] = {}
_DTE_CAPACITY_TB = []
for _nk in x:
    _candidates = [
        p
        for p in _pkl_files
        if os.path.basename(p).split(".")[0].split("_")[-1] == str(_nk)
    ]
    if _candidates:
        with open(_candidates[0], "rb") as _f:
            _teg = pickle.load(_f)
        _gs = sum(1 for n in _teg.nodes if n.id in _dest_ids)
        _src = sum(1 for n in _teg.nodes if n.id in _source_ids)
        _rly = sum(1 for n in _teg.nodes if n.id in _relay_ids)
        node_count_to_label[_nk] = f"{_src}/{_rly}"
        # _GS_COUNT ground stations × 86400 s/day × source bit rate (1000 internal units = 267 Mbps)
        _DTE_CAPACITY_TB.append(
            _gs * 86400 * 1000 * _BITS_PER_INTERNAL_UNIT / _BITS_PER_TERABIT
        )
        del _teg
    else:
        node_count_to_label[_nk] = str(_nk)

plot_dir = os.path.join(PLOTS_ROOT, str(report_id))
pdf_dir = os.path.join(plot_dir, "pdf")
png_dir = os.path.join(plot_dir, "png")
os.makedirs(pdf_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

for metric, unit, y_min, y_max, y_step in metrics:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for algorithm, display_name, linestyle, linewidth, color in algorithms:
        y = [run[metric] for run in report if run["Algorithm"] == algorithm]
        if not y:
            continue
        kwargs = dict(
            label=display_name, linewidth=linewidth, linestyle=linestyle
        )
        if color:
            kwargs["color"] = color
        plt.plot(x[: len(y)], y, **kwargs)

    if metric == "Capacity":
        plt.plot(
            x,
            _DTE_CAPACITY_TB,
            label="DTE Capacity",
            linewidth=2.5,
            color="gold",
            linestyle="dashed",
        )

    if metric == "Capacity by node":
        plt.plot(
            x,
            [_DTE_CAPACITY_TB[i] / n for i, n in enumerate(x)],
            label="DTE Capacity",
            linewidth=2.5,
            color="gold",
            linestyle="dashed",
        )

    label = f"{metric} [{unit}]" if unit else metric
    plt.ylabel(f"Delay [{unit}]" if metric == "Scheduled delay" else label)
    plt.xlabel("Source/relay node counts")
    plt.legend()
    plt.grid(linestyle="-", color="0.95")

    if metric == "Execution duration":
        plt.yscale("log")
    elif metric == "Jain's fairness index":
        plt.ylim(y_min, y_max)
    else:
        ax.relim()
        ax.autoscale_view()
        _, auto_top = ax.get_ylim()
        top = (math.floor(auto_top / y_step) + 1) * y_step
        plt.ylim(max(y_min - y_step, 0), top)

    # Thin ticks when there are too many x values (keep ≤ 12)
    x_ticks = x if len(x) <= 12 else x[:: math.ceil(len(x) / 12)]
    x_labels = [node_count_to_label.get(v, str(v)) for v in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    file_name = label.replace(" ", "_").replace("/", "_")
    plt.savefig(
        os.path.join(pdf_dir, f"{file_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join(png_dir, f"{file_name}.png"),
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
