import math
import os
import pickle
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from laser_link_scheduler.graph.time_expanded_graph import TimeExpandedGraph
from laser_link_scheduler.topology.weights import compute_delays


plt.rcParams.update({"font.size": 18})
plt.rc("legend", fontsize=14)
plt.rcParams.update({"font.family": "Times New Roman"})

algorithms = ["lls", "lls_pat_unaware", "lls_mip", "fcp"]

report_id = 1748949730
tegs = []

pattern = re.compile(r"^([a-zA-Z_]+)_gs_.*?_(\d+)\.pkl$")

report_dir = os.path.join("reports", str(report_id))
for file_name in os.listdir(report_dir):
    if file_name.endswith(".pkl"):
        file_path = os.path.join(report_dir, file_name)

        match = pattern.match(file_name)
        algorithm = match.group(1)
        number = int(match.group(2))

        if algorithm not in algorithms:
            continue

        with open(file_path, "rb") as f:
            teg: TimeExpandedGraph = pickle.load(f)
            tegs.append((algorithm, number, teg))

all_pointing_delays = []
all_link_acq_delays = []

retargeting_duty_cycles = []

for algorithm, node_count, teg in tegs:
    print(f"Processing {algorithm} node count {node_count}")
    delays_by_node = {node: [] for node in teg.nodes}

    for k in range(teg.K):
        for tx_oi_idx in range(teg.N):
            for rx_oi_idx in range(teg.N):
                if teg.graphs[k][tx_oi_idx][rx_oi_idx] == 1:
                    pointing_delay, link_acq_delay = compute_delays(
                        tx_oi_idx,
                        rx_oi_idx,
                        teg.graphs[:k],
                        teg.state_durations[k],
                        teg.pos,
                        teg.optical_interfaces_to_node,
                        teg.nodes,
                    )

                    tx_node = teg.nodes[
                        teg.optical_interfaces_to_node[tx_oi_idx]
                    ]
                    rx_node = teg.nodes[
                        teg.optical_interfaces_to_node[rx_oi_idx]
                    ]

                    # state duration, pointing delay, link acq delay
                    delays_by_node[tx_node].append(
                        (
                            teg.state_durations[k],
                            pointing_delay,
                            link_acq_delay,
                        )
                    )
                    delays_by_node[rx_node].append(
                        (
                            teg.state_durations[k],
                            pointing_delay,
                            link_acq_delay,
                        )
                    )

                    all_pointing_delays.append(pointing_delay)
                    all_link_acq_delays.append(link_acq_delay)

    network_total_time = 0
    network_total_eff_time = 0
    retargeting_duty_cycle = {}
    for node, delays in delays_by_node.items():
        total_time = 0
        total_eff_time = 0
        for state_duration, pointing_delay, link_acq_delay in delays:
            total_time += state_duration
            total_eff_time += state_duration - (
                pointing_delay + link_acq_delay
            )

            network_total_time += state_duration
            network_total_eff_time += state_duration - (
                pointing_delay + link_acq_delay
            )

        # proportion of time spent transmitting vs total time
        retargeting_duty_cycle[node] = total_eff_time / total_time
        # print(node, total_time, total_eff_time)

    # pprint.pprint(retargeting_duty_cycle)

    network_retargeting_duty_cycle = (
        network_total_eff_time / network_total_time
    )
    print("network retargeting duty cycle", network_retargeting_duty_cycle)
    retargeting_duty_cycles.append(
        (algorithm, node_count, network_retargeting_duty_cycle)
    )

algorithms = [
    ("lls", "LLS_Greedy"),
    ("lls_pat_unaware", "LLS_Greedy (ZRK)"),
    ("lls_mip", "LLS_MIP"),
    ("fcp", "FCP"),
]

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
os.makedirs("analysis", exist_ok=True)
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
