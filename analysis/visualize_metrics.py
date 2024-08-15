import os
import csv
import pprint

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 18})
plt.rc('legend', fontsize=14)
plt.rcParams.update({'font.family': 'Times New Roman'})

report_id = 1723687797
path = os.path.join("reports", str(report_id), f"{report_id}_report.csv")
with open(path, "r") as f:
    report = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    
for run in report:
    run["Capacity"] = int(run["Capacity"]) * 267_000 / 1000 / 1000 / 1000 / 1000
    run["Wasted capacity"] = int(run["Wasted capacity"]) * 267_000 / 1000 / 1000 / 1000 / 1000
    run["Scheduled delay"] = float(run["Scheduled delay"])
    run["Jain's fairness index"] = float(run["Jain's fairness index"])

# pprint.pprint(report)

display_name = {
    "lls": "LLS",
    "fcp": "FCP",
    "random": "Random",
}

metrics = [
    ("Capacity", "terabits", 20, 200, 20),
    ("Wasted capacity", "terabits", 40, 280, 40),
    ("Scheduled delay", "seconds", 2_000, 10_000, 2_000),
    ("Jain's fairness index", "", 0.9, 1.0, 0.02),
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

    for algorithm in ["fcp", "random", "lls"]:
        y = [run[metric] for run in report if run["Algorithm"] == algorithm]

        plt.plot(x, y, label=display_name[algorithm], linewidth=2.5)

    label = f"{metric} [{unit}]" if unit else metric
    plt.ylabel(label)
    plt.xlabel("Network Size")
    plt.legend()

    plt.grid(linestyle='-', color='0.95')
    
    plt.ylim(max(y_min-y_step, 0), y_max)
    ax.set_yticks(np.arange(y_min, y_max+0.01, y_step))

    file_name = label.replace(" ", "_")
    plt.savefig(
        os.path.join("analysis", f"{file_name}.pdf"),
        format="pdf",
        bbox_inches='tight'
    )
    plt.savefig(
        os.path.join("analysis", f"{file_name}.png"),
        format="png",
        bbox_inches='tight'
    )
    # plt.show()
