import os
import csv
import math
import pprint

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=16)
plt.rcParams.update({'font.family': 'Times New Roman'})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'pdf.fonttype': 42})

report_id = 1774132452
path = os.path.join("reports", str(report_id), f"{report_id}_report.csv")
with open(path, "r") as f:
    report = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    
scenarios = []
for run in report:
    num_nodes = int(run["Scenario"].split("_")[-1])
    scenarios.append(num_nodes)

    run["Capacity by node"] = float(run["Capacity"]) / 1000/ 1000 / num_nodes  # this has to come before capacity calculation
    run["Capacity"] = float(run["Capacity"]) / 1000 / 1000 / 1000
    run["Wasted capacity"] = float(run["Wasted capacity"]) / 1000 / 1000 / 1000
    run["Wasted buffer"] = float(run["Wasted buffer capacity"]) / 1000 / 1000 / 1000
    run["Scheduled delay"] = float(run["Scheduled delay"]) / 60 / 60
    run["Jain's fairness index"] = float(run["Jain's fairness index"])
    run["Execution duration"] = float(run["Execution duration"])

x = sorted(list(set(scenarios)))

# pprint.pprint(report)

algorithms = [
    ("lls", "LLS_Greedy"),
    ("lls_pat_unaware", "LLS_Greedy (ZRK)"),
    ("lls_mip", "LLS_MIP"),
    ("fcp", "FCP"),
]

metrics = [
    ("Capacity", "petabits/day", 0, 40, 5),
    ("Capacity by node", "terabits/day", 0, 500, 100),
    ("Wasted capacity", "petabits/day", 5, 40, 5),
    ("Wasted buffer", "petabits/day", 5, 40, 5),
    ("Scheduled delay", "hours", 0, 20, 4),
    ("Jain's fairness index", "", 0.5, 1.0, 0.1),
    ("Execution duration", "seconds", 0.1, 100000, 30),
]

OUTPUT_DIR = "sparse_optical_earth_network_scenario_analysis"

for metric, unit, y_min, y_max, y_step in metrics:
    print(f"Processing metric {metric}")
    fig, ax = plt.subplots()

    for algorithm, display_name in algorithms:
        y = [run[metric] for run in report if run["Algorithm"] == algorithm]

        # if algorithm == "lls_lp":
        #     plt.plot(x[:len(y)], y, linestyle="dashed", label=display_name, linewidth=3.5)
        # elif algorithm == "lls_mip":
        #     plt.plot(x[:len(y)], y, linestyle="dotted", label=display_name, linewidth=3.5)
        # else:
        #     plt.plot(x[:len(y)], y, label=display_name, linewidth=2.5)

        ax.plot(x[:len(y)], y, label=display_name, linewidth=2.5)
    
    label = f"{metric} [{unit}]" if unit else metric
    if metric == "Scheduled delay":
        ax.set_ylabel(f"Delay [{unit}]")
    else:
        ax.set_ylabel(label)

    ax.set_xlabel("LEO node count")
    ax.legend()

    ax.grid(linestyle='-', color='0.95')

    if metric == "Execution duration":
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
    elif metric == "Jain's fairness index":
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(max(y_min - y_step, 0), y_max)
        ax.set_yticks([y_min] + np.arange(y_step, y_max + 0.01, y_step).tolist())

    ax.set_xticks([32, 64, 128, 192, 264])
    # ax.set_xticklabels([
    #     "32/6/4",
    #     "64/9/6",
    #     "128/12/8",
    #     "192/15/10",
    #     "264/18/12",
    # ])

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        ncol=2
    )

    bbox = dict(boxstyle="round", fc="0.9")
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=90,rad=10")

    plt.axvline(
        x=64.0,
        color='black',
        linestyle='dashed',
        linewidth=2,
        # zorder=0,
    )
    plt.text(
        110.0,
        y_max * 0.8 if metric != "Execution duration" else y_max * 0.5,
        "MILP model\nintractable",
        fontsize=16,
        ha='center',
        va='top',
        bbox=bbox
    )
    
    file_name = label.replace(" ", "_").replace("/", "_").replace("[", "").replace("]", "").replace("'", "")
    
    # plt.tight_layout()

    plt.savefig(
        os.path.join("analysis", OUTPUT_DIR, f"{file_name}.pdf"),
        format="pdf",
        bbox_inches="tight"
    )
    plt.savefig(
        os.path.join("analysis", OUTPUT_DIR, f"{file_name}.png"),
        format="png",
        bbox_inches="tight",
        dpi=300,
    )
    # plt.show()
