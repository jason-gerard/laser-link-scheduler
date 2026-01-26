import csv
import os

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

report_id = 1739218035
path = os.path.join("reports", str(report_id), f"{report_id}_report.csv")
with open(path, "r") as f:
    report = [
        {k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)
    ]

for run in report:
    run["Capacity"] = (
        float(run["Capacity"]) * 267_000 / 1000 / 1000 / 1000 / 1000
    )
    run["Wasted capacity"] = (
        float(run["Wasted capacity"]) * 267_000 / 1000 / 1000 / 1000 / 1000
    )
    run["Wasted buffer capacity"] = (
        float(run["Wasted buffer capacity"])
        * 267_000
        / 1000
        / 1000
        / 1000
        / 1000
    )
    run["Scheduled delay"] = float(run["Scheduled delay"]) / 60
    run["Jain's fairness index"] = float(run["Jain's fairness index"])
    run["Execution duration"] = float(run["Execution duration"])

# pprint.pprint(report)

algorithms = [
    ("lls", "LLS_Greedy"),
    ("lls_pat_unaware", "LLS_PAT_Unaware"),
    ("lls_mip", "LLS_MIP"),
    ("lls_lp", "LLS_LP"),
    ("fcp", "FCP"),
    ("random", "Random"),
    ("alternating", "Alternating"),
]

metrics = [
    ("Capacity", "terabits/day", 40, 280, 40),
    ("Wasted capacity", "terabits/day", 20, 100, 20),
    ("Wasted buffer capacity", "terabits/day", 5, 40, 5),
    ("Scheduled delay", "minutes", 20, 200, 20),
    ("Jain's fairness index", "", 0.2, 1.0, 0.2),
    ("Execution duration", "seconds", 0.01, 100000, 30),
]

x = [
    "X-Small",
    "Small",
    "Medium",
    "Large",
    "X-Large",
]

for metric, unit, y_min, y_max, y_step in metrics:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for algorithm, display_name in algorithms:
        y = [run[metric] for run in report if run["Algorithm"] == algorithm]

        plt.plot(x[: len(y)], y, label=display_name, linewidth=2.5)

    label = f"{metric} [{unit}]" if unit else metric
    plt.ylabel(label)
    plt.xlabel("Network Size")
    plt.legend()

    plt.grid(linestyle="-", color="0.95")

    if metric == "Execution duration":
        plt.yscale("log")
        plt.ylim(y_min, y_max)
    else:
        plt.ylim(max(y_min - y_step, 0), y_max)
        ax.set_yticks(
            [y_min] + np.arange(y_step, y_max + 0.01, y_step).tolist()
        )

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
    # plt.show()
