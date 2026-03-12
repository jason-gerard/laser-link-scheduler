import math
import os
from pathlib import Path
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.scripts.utils import compute_energy_metrics  # noqa: E402
from src.time_expanded_graph.time_expanded_graph import (  # noqa: E402
    TimeExpandedGraph,
)


plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

algorithms = [
    ("lls", "LLS_Greedy"),
    ("lls_pat_unaware", "LLS_Greedy (ZRK)"),
    ("lls_mip", "LLS_MIP"),
    ("fcp", "FCP"),
]

report_id = 1773312306
tegs = []

pattern = re.compile(r"^([a-zA-Z_]+).*?_(\d+)\.pkl$")

report_dir = os.path.join("output", "reports", str(report_id))
for file_name in os.listdir(report_dir):
    if file_name.endswith(".pkl"):
        file_path = os.path.join(report_dir, file_name)

        match = file_name.split(("_"))
        if match is None:
            print(f"{file_name} doesn't match with the pattern")
            continue
        algorithm = match[0]
        number = int(match[-1].split(".")[0])

        if algorithm not in [alg[0] for alg in algorithms]:
            continue
        with open(file_path, "rb") as f:
            teg: TimeExpandedGraph = pickle.load(f)
            tegs.append((algorithm, number, teg))

all_generation = []
all_consumption = []
for algorithm, node_count, teg in tegs:
    print(f"Processing {algorithm} node count {node_count}")
    metrics_by_node = {node: [] for node in teg.nodes}
    acumulated_time = 0
    for k in range(teg.K):
        for tx_oi_idx in range(teg.N):
            for rx_oi_idx in range(teg.N):
                if teg.graphs[k][tx_oi_idx][rx_oi_idx] == 1:
                    tx_node_id = teg.nodes[
                        teg.optical_interfaces_to_node[tx_oi_idx]
                    ].id
                    rx_node_id = teg.nodes[
                        teg.optical_interfaces_to_node[rx_oi_idx]
                    ].id

                    generated, consumed = compute_energy_metrics(
                        teg,
                        tx_oi_idx,
                        rx_oi_idx,
                        teg.graphs[:k],
                        teg.state_durations[k],
                        acumulated_time,
                        tx_node_id,
                    )

                    # state duration, generated energy, consumed energy
                    metrics_by_node[tx_node_id].append(
                        (
                            teg.state_durations[k],
                            generated,
                            consumed,
                        )
                    )
                    metrics_by_node[rx_node_id].append(
                        (
                            teg.state_durations[k],
                            generated,
                            consumed,
                        )
                    )

                    all_generation.append(generated)
                    all_consumption.append(consumed)
        # move the state time pointer
        acumulated_time += teg.state_durations[k]

    network_total_time = 0
    network_total_additional_energy = 0
    network_total_generated = 0
    network_total_consumed = 0

    state_additional_energy = {}
    for node_id, metrics in metrics_by_node.items():
        total_time = 0
        total_additional_energy = 0
        total_generated = 0
        total_consumed = 0
        for state_duration, generated, consumed in metrics:
            total_time += state_duration
            total_generated += generated
            total_consumed += consumed

            network_total_time += state_duration
            network_total_additional_energy += generated - consumed

        # proportion of energy spent transmitting vs total generated
        # retargeting_duty_cycle[node_id] = total_eff_time / total_time
        # print(node, total_time, total_eff_time)

    # pprint.pprint(retargeting_duty_cycle)

    # network_retargeting_duty_cycle = (
    #     network_total_eff_time / network_total_time
    # )
    # print("network retargeting duty cycle", network_retargeting_duty_cycle)
    # retargeting_duty_cycles.append(
    #     (algorithm, node_count, network_retargeting_duty_cycle)
    # )


retargeting_duty_cycles = []
# X-axis ticks
x = sorted(
    list(set([node_count for _, node_count, _ in retargeting_duty_cycles]))
)

# Plot setup
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot for each algorithm
for algorithm, display_name in algorithms:
    y = [
        duty
        for (alg, node_count, duty) in retargeting_duty_cycles
        if alg == algorithm
    ]
    x_vals = [
        node_count
        for (alg, node_count, duty) in retargeting_duty_cycles
        if alg == algorithm
    ]

    # Sort by x for proper line plotting
    sorted_pairs = sorted(zip(x_vals, y))
    x_sorted, y_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])

    if algorithm == "lls_mip":
        plt.plot(
            x_sorted,
            y_sorted,
            linestyle="dotted",
            label=display_name,
            linewidth=3.5,
        )
    else:
        plt.plot(x_sorted, y_sorted, label=display_name, linewidth=2.5)

# Labels and formatting
plt.ylabel("Retargeting Duty Cycle [%]")
plt.xlabel("Source/Relay Node Count")
plt.ylim(0.6, 1.0)
plt.yticks(np.arange(0.6, 1.01, 0.05))
plt.grid(linestyle="-", color="0.95")
plt.legend(loc="lower right")

# Custom X-axis ticks and labels
ax.set_xticks([i for i in x if i % 8 == 0])
ax.set_xticklabels([f"{i}/{math.ceil(i / 16)}" for i in x if i % 8 == 0])

# Save the figure
file_name = "network_retargeting_duty_cycle"
os.makedirs("output/plots", exist_ok=True)
plt.savefig(
    os.path.join("output/plots", f"{file_name}.pdf"),
    format="pdf",
    bbox_inches="tight",
)
plt.savefig(
    os.path.join("output/plots", f"{file_name}.png"),
    format="png",
    bbox_inches="tight",
    dpi=300,
)
