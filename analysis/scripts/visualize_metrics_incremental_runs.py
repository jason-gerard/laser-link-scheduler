import csv
import math
import os

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

report_id = 1748949730
path = os.path.join("reports", str(report_id), f"{report_id}_report.csv")
with open(path, "r") as f:
    report = [
        {k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)
    ]

scenarios = []
for run in report:
    num_nodes = int(run["Scenario"].split("_")[-1])
    scenarios.append(num_nodes)

    run["Capacity by node"] = (
        float(run["Capacity"])
        * 267_000
        * 4
        / 1000
        / 1000
        / 1000
        / 1000
        / num_nodes
    )  # this has to come before capacity calculation
    run["Capacity"] = (
        float(run["Capacity"]) * 267_000 * 4 / 1000 / 1000 / 1000 / 1000
    )
    run["Wasted capacity"] = (
        float(run["Wasted capacity"]) * 267_000 * 4 / 1000 / 1000 / 1000 / 1000
    )
    run["Wasted buffer capacity"] = (
        float(run["Wasted buffer capacity"])
        * 267_000
        / 1000
        / 1000
        / 1000
        / 1000
    )
    run["Scheduled delay"] = float(run["Scheduled delay"]) / 60 / 60
    run["Jain's fairness index"] = float(run["Jain's fairness index"])
    run["Execution duration"] = float(run["Execution duration"])

x = sorted(list(set(scenarios)))

# pprint.pprint(report)

algorithms = [
    ("lls", "LLS_Greedy"),
    ("lls_pat_unaware", "LLS_Greedy (ZRK)"),
    # ("lls_lp", "LLS_LP"),
    ("lls_mip", "LLS_MIP"),
    ("fcp", "FCP"),
    # ("random", "Random"),
    # ("alternating", "Alternating"),
]

metrics = [
    ("Capacity", "terabits/day", 5, 40, 5),
    ("Capacity by node", "terabits/day", 0.5, 4, 0.5),
    ("Wasted capacity", "terabits/day", 10, 70, 10),
    ("Wasted buffer capacity", "terabits/day", 5, 40, 5),
    ("Scheduled delay", "hours", 0, 6, 1),
    ("Jain's fairness index", "", 0.5, 1.0, 0.2),
    ("Execution duration", "seconds", 0.01, 100000, 30),
]

for metric, unit, y_min, y_max, y_step in metrics:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for algorithm, display_name in algorithms:
        y = [run[metric] for run in report if run["Algorithm"] == algorithm]

        if algorithm == "lls_lp":
            plt.plot(
                x[: len(y)],
                y,
                linestyle="dashed",
                label=display_name,
                linewidth=3.5,
            )
        elif algorithm == "lls_mip":
            plt.plot(
                x[: len(y)],
                y,
                linestyle="dotted",
                label=display_name,
                linewidth=3.5,
            )
        else:
            plt.plot(x[: len(y)], y, label=display_name, linewidth=2.5)

    if metric == "Capacity":
        # bbox = dict(boxstyle="round", fc="0.9")
        # arrowprops = dict(
        #     arrowstyle="->",
        #     connectionstyle="angle,angleA=0,angleB=90,rad=10")
        # ax.annotate("LLS_MIP intractable\nbeyond this point", fontsize=13, xy=(48, 30),
        #             # xytext=(-102, 24), textcoords='offset points',
        #             xytext=(-51, -150), textcoords='offset points',
        #             bbox=bbox, arrowprops=arrowprops)

        # num_gs * duration * deep space data rate
        y = [
            3 * 86400 * 187 * 267_000 / 1000 / 1000 / 1000 / 1000
            for _ in range(len(x))
        ]
        plt.plot(
            x,
            y,
            label="DTE Capacity",
            linewidth=2.5,
            color="gold",
            linestyle="dashed",
        )

    if metric == "Capacity by node":
        # num_gs * duration * deep space data rate
        y = [
            3 * 86400 * 187 * 267_000 / 1000 / 1000 / 1000 / 1000 / x
            for x in x
        ]
        plt.plot(
            x,
            y,
            label="DTE Capacity",
            linewidth=2.5,
            color="gold",
            linestyle="dashed",
        )

    label = f"{metric} [{unit}]" if unit else metric
    if metric == "Scheduled delay":
        plt.ylabel(f"Delay [{unit}]")
    else:
        plt.ylabel(label)
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
