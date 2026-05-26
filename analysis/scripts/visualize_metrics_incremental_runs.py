import csv
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import REPORTS_ROOT, PLOTS_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.constants import REPO_ROOT, REPORTS_ROOT, PLOTS_ROOT

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

# Algorithms: baselines + energy/lifetime-aware contributions
algorithms = [
    # Baselines
    ("lls", "LLS_Greedy", "solid", 2.5, None),
    ("lls_pat_unaware", "LLS_Greedy (ZRK)", "dashed", 2.5, None),
    ("fcp", "FCP", "solid", 2.5, None),
    # Contributions
    ("energy_aware", "Energy-Aware", "solid", 2.5, None),
    ("battery_energy", "Battery-Aware", "dashed", 2.5, None),
    ("lifespan_aware", "Lifespan-Aware", "dotted", 2.5, None),
]

metrics = [
    ("Capacity", "terabits/day", 0, 40, 5),
    ("Capacity by node", "terabits/day", 0, 4, 0.5),
    ("Wasted capacity", "terabits/day", 0, 70, 10),
    ("Wasted buffer capacity", "terabits/day", 0, 40, 5),
    ("Scheduled delay", "hours", 0, 6, 1),
    ("Jain's fairness index", "", 0.5, 1.0, 0.1),
    ("Execution duration", "seconds", 0.01, 1e5, 30),
]

# DTE capacity baseline: 3 ground stations × 86400 s/day × source bit rate (1000 internal units = 267 Mbps)
_DTE_CAPACITY_TB = (
    3 * 86400 * 1000 * _BITS_PER_INTERNAL_UNIT / _BITS_PER_TERABIT
)

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
            [_DTE_CAPACITY_TB] * len(x),
            label="DTE Capacity",
            linewidth=2.5,
            color="gold",
            linestyle="dashed",
        )

    if metric == "Capacity by node":
        plt.plot(
            x,
            [_DTE_CAPACITY_TB / n for n in x],
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
        plt.ylim(y_min, y_max)
    elif metric == "Jain's fairness index":
        plt.ylim(y_min, y_max)
    else:
        plt.ylim(max(y_min - y_step, 0), y_max)
        ax.set_yticks(
            [y_min] + np.arange(y_step, y_max + 0.01, y_step).tolist()
        )

    ax.set_xticks([i for i in x if i % 8 == 0])
    ax.set_xticklabels([f"{i}/{math.ceil(i / 16)}" for i in x if i % 8 == 0])

    file_name = label.replace(" ", "_").replace("/", "_")
    plt.savefig(
        os.path.join(PLOTS_ROOT, f"{file_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        os.path.join("analysis", f"{file_name}.png"),
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)
