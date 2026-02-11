import os
import csv
import math

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=16)
plt.rcParams.update({'font.family': 'Times New Roman'})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'pdf.fonttype': 42})

report_id = 1770632046
path = os.path.join("reports", str(report_id), f"{report_id}_report.csv")
with open(path, "r") as f:
    report = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    
scenarios = []
for run in report:
    num_nodes = int(run["Scenario"].split("_")[-1])
    scenarios.append(num_nodes)

    run["Capacity by node"] = float(run["Capacity"]) / 1000 / 1000 / num_nodes  # this has to come before capacity calculation
    run["Capacity"] = float(run["Capacity"]) / 1000 / 1000
    run["Wasted capacity"] = float(run["Wasted capacity"]) / 1000 / 1000
    run["Wasted buffer capacity"] = float(run["Wasted buffer capacity"]) / 1000 / 1000
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
    ("Capacity", "terabits/day", 5, 55, 5),
    ("Capacity by node", "terabits/day", 0.5, 3.5, 0.5),
    ("Wasted capacity", "terabits/day", 0, 30, 5),
    ("Wasted buffer capacity", "terabits/day", 0, 25, 5),
    ("Scheduled delay", "hours", 0, 20, 2),
    ("Jain's fairness index", "", 0.8, 1.0, 0.2),
    ("Execution duration", "seconds", 0.01, 1000000, 30),
]

OGS_BIT_RATE_TB = 50 / 1000 / 1000
OUTPUT_DIR = "mars_relay_earth_scenario_analysis"

for metric, unit, y_min, y_max, y_step in metrics:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for algorithm, display_name in algorithms:
        y = [run[metric] for run in report if run["Algorithm"] == algorithm]
        plt.plot(x[:len(y)], y, label=display_name, linewidth=2.5)
    
    if metric == "Capacity":
        y = [math.ceil(x[i]/16) * 3 * 86400 * OGS_BIT_RATE_TB for i in range(len(x))]
        plt.plot(x, y, label="DTE Capacity", linewidth=2.5, color="gold", linestyle="dashed")

    if metric == "Capacity by node":
        y = [math.ceil(x/16) * 3 * 86400 * OGS_BIT_RATE_TB / x for x in x]
        plt.plot(x, y, label="DTE Capacity", linewidth=2.5, color="gold", linestyle="dashed")

    label = f"{metric} [{unit}]" if unit else metric
    if metric == "Scheduled delay":
        plt.ylabel(f"Delay [{unit}]")
    else:
        plt.ylabel(label)
    plt.xlabel("Source/relay node counts")
    plt.legend()

    plt.grid(linestyle='-', color='0.95')

    if metric == "Execution duration":
        plt.yscale("log")
        plt.ylim(y_min, y_max)
    elif metric == "Jain's fairness index":
        plt.ylim(y_min, y_max)
    else:
        plt.ylim(max(y_min-y_step, 0), y_max)
        ax.set_yticks([y_min] + np.arange(y_step, y_max+0.01, y_step).tolist())

    ax.set_xticks([i for i in x if i % 16 == 0])
    ax.set_xticklabels([f"{i}/{3 * math.ceil(i/16)}" for i in x if i % 16 == 0])
    
    file_name = label.replace(" ", "_").replace("/", "_").replace("[", "").replace("]", "").replace("'", "")
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
