import os
import csv
import pprint

import matplotlib.pyplot as plt

report_id = 1723687797
path = os.path.join("reports", str(report_id), f"{report_id}_report.csv")
with open(path, "r") as f:
    report = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    
for run in report:
    run["Capacity"] = int(run["Capacity"]) / 8 / 1000 / 1000
    run["Wasted capacity"] = int(run["Wasted capacity"]) / 8 / 1000 / 1000
    run["Scheduled delay"] = float(run["Scheduled delay"])
    run["Jain's fairness index"] = float(run["Jain's fairness index"])

pprint.pprint(report)

metrics = [
    ("Capacity", "megabytes"),
    ("Wasted capacity", "megabytes"),
    ("Scheduled delay", "seconds"),
    ("Jain's fairness index", ""),
]

x = [
    "X-Small",
    "Small",
    "Medium",
    "Large",
    "X-Large",
]

for metric, unit in metrics:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for algorithm in ["fcp", "random", "lls"]:
        y = [run[metric] for run in report if run["Algorithm"] == algorithm]

        plt.plot(x, y, label=algorithm)

    label = f"{metric} [{unit}]" if unit else metric
    plt.ylabel(label)
    plt.xlabel("Network Size")
    plt.legend()

    plt.grid(linestyle='-', color='0.95')
    
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
