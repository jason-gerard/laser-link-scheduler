import os
import pickle
import pprint
import sys
import re
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(os.path.join(os.path.dirname(sys.path[0]), ".."))
from time_expanded_graph import TimeExpandedGraph
from weights import compute_delays

plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=16)
plt.rcParams.update({'font.family': 'Times New Roman'})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'pdf.fonttype': 42})

algorithms = [
    ("otls", "OTLS_Greedy"),
    ("otls_pat_unaware", "OTLS_Greedy (ZRK)"),
    ("otls_mip", "OTLS_MIP"),
    ("lls", "LLS_Greedy"),
    ("lls_pat_unaware", "LLS_Greedy (ZRK)"),
    ("lls_mip", "LLS_MIP"),
    ("fcp", "FCP"),
]

report_id = 1770805980
# OUTPUT_DIR = "mars_relay_earth_scenario_analysis"
OUTPUT_DIR = "mars_earth_relay_scenario_analysis"
# OUTPUT_DIR = "mars_earth_dte_scenario_analysis"

tegs = []

pattern = re.compile(r"^([a-zA-Z_]+)_mars_.*?_(\d+)\.pkl$")
# pattern = re.compile(r"^([a-zA-Z_]+)_gs_.*?_(\d+)\.pkl$")
# pattern = re.compile(r"^([a-zA-Z_]+)_dte_.*?_(\d+)\.pkl$")

report_dir = os.path.join("reports", str(report_id))
for file_name in os.listdir(report_dir):
    if file_name.endswith(".pkl"):
        file_path = os.path.join(report_dir, file_name)

        match = pattern.match(file_name)
        algorithm = match.group(1)
        number = int(match.group(2))

        if algorithm not in [algorithm[0] for algorithm in algorithms]:
            print(f"Skipping algo {algorithm}")
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
                
                    tx_node = teg.nodes[teg.optical_interfaces_to_node[tx_oi_idx]]
                    rx_node = teg.nodes[teg.optical_interfaces_to_node[rx_oi_idx]]
                    
                    # state duration, pointing delay, link acq delay
                    delays_by_node[tx_node].append((teg.state_durations[k], pointing_delay, link_acq_delay))
                    delays_by_node[rx_node].append((teg.state_durations[k], pointing_delay, link_acq_delay))
                    
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
            total_eff_time += state_duration - (pointing_delay + link_acq_delay)

            network_total_time += state_duration
            network_total_eff_time += state_duration - (pointing_delay + link_acq_delay)

        # proportion of time spent transmitting vs total time
        if total_time == 0:
            # print(node, total_time, total_eff_time)
            retargeting_duty_cycle[node] = 0
        else:
            retargeting_duty_cycle[node] = total_eff_time / total_time

    # pprint.pprint(retargeting_duty_cycle)

    network_retargeting_duty_cycle = 100 * (network_total_eff_time / network_total_time)
    print("network retargeting duty cycle", network_retargeting_duty_cycle)
    retargeting_duty_cycles.append((algorithm, node_count, network_retargeting_duty_cycle))

# X-axis ticks
x = sorted(list(set([node_count for _, node_count, _ in retargeting_duty_cycles])))

# Plot setup
fig = plt.figure()
ax = fig.add_subplot(111)

# Plot for each algorithm
for algorithm, display_name in algorithms:
    y = [
        duty for (alg, node_count, duty) in retargeting_duty_cycles
        if alg == algorithm
    ]
    x_vals = [
        node_count for (alg, node_count, duty) in retargeting_duty_cycles
        if alg == algorithm
    ]
    
    # Sort by x for proper line plotting
    sorted_pairs = sorted(zip(x_vals, y))
    x_sorted, y_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])

    # if algorithm == "lls_mip":
    #     plt.plot(x_sorted, y_sorted, linestyle="dotted", label=display_name, linewidth=3.5)
    # else:
    #     plt.plot(x_sorted, y_sorted, label=display_name, linewidth=2.5)

    plt.plot(x_sorted, y_sorted, label=display_name, linewidth=2.5)

# Labels and formatting
plt.ylabel("Network Duty Cycle [%]")
plt.xlabel("Source/Relay Node Count")
plt.ylim(60, 100)
plt.yticks(np.arange(60, 101, 5))
plt.grid(linestyle='-', color='0.95')
plt.legend(loc="lower right")

# Custom X-axis ticks and labels
# ax.set_xticks([i for i in x if i % 16 == 0])
ax.set_xticks([i for i in x if i % 8 == 0])
# ax.set_xticklabels([f"{i}/{3 * math.ceil(i/16)}" for i in x if i % 16 == 0])
ax.set_xticklabels([f"{i}/{math.ceil(i/8)}" for i in x if i % 8 == 0])
# ax.set_xticklabels([f"{i}/{math.ceil(i/16)}" for i in x if i % 16 == 0])

# Save the figure
file_name = "network_retargeting_duty_cycle"
plt.savefig(os.path.join("analysis", OUTPUT_DIR, f"{file_name}.pdf"), format="pdf", bbox_inches="tight")
plt.savefig(os.path.join("analysis", OUTPUT_DIR, f"{file_name}.png"), format="png", bbox_inches="tight", dpi=300)
