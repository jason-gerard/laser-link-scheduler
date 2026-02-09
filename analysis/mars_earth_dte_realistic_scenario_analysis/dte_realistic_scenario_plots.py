import os
import pickle
import pprint
import sys
import re
import math
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import numpy as np
from scipy.stats import gaussian_kde

import matplotlib as mpl
from matplotlib.colors import ListedColormap

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from time_expanded_graph import TimeExpandedGraph
from weights import compute_node_capacities, compute_capacity

plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=14)
plt.rcParams.update({'font.family': 'Times New Roman'})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'pdf.fonttype': 42})

algorithms = [
    ("otls", "OTLS_Greedy"),
    ("otls_pat_unaware", "OTLS_Greedy (ZRK)"),
    ("fcp", "FCP"),
]

report_id = 1770075847
tegs = []

pattern = re.compile(r"^([a-zA-Z_]+)_dte_.*?.pkl$")

report_dir = os.path.join("reports", str(report_id))
for file_name in os.listdir(report_dir):
    if file_name.endswith(".pkl"):
        file_path = os.path.join(report_dir, file_name)

        match = pattern.match(file_name)
        algorithm = match.group(1)

        if algorithm not in [algo[0] for algo in algorithms]:
            continue

        with open(file_path, "rb") as f:
            teg: TimeExpandedGraph = pickle.load(f)
            tegs.append((algorithm, teg))

capacities_by_algo = {algo: [] for algo in [algo[0] for algo in algorithms]}
detla_capacities_by_algo = {algo: [] for algo in [algo[0] for algo in algorithms]}

for algorithm, teg in tegs:
    print(f"Processing {algorithm}, K = {teg.K}...")
    for k in tqdm(range(teg.K)):
        if k == 0:
            capacities_by_algo[algorithm].append(0)

        node_capacities = compute_node_capacities(
            teg.graphs,
            teg.state_durations,
            k,
            teg.nodes,
            teg.graphs,
            teg.pos,
            teg.optical_interfaces_to_node,
            teg.node_to_optical_interfaces
        )
        network_capacity = int(compute_capacity(node_capacities))
        capacities_by_algo[algorithm].append(network_capacity / 1000) # mb to gb
    
    # transform to delta caps
    for index, cap in enumerate(capacities_by_algo[algorithm]):
        if index == 0:
            detla_capacities_by_algo[algorithm].append(capacities_by_algo[algorithm][0])
        else:
            detla_capacities_by_algo[algorithm].append(capacities_by_algo[algorithm][index] - capacities_by_algo[algorithm][index-1])

# print(capacities_by_algo)
# print(detla_capacities_by_algo)

# Plot setup
fig = plt.figure(figsize=(12, 3))
ax = fig.add_subplot(111)

# Plot for each algorithm
for algorithm, display_name in algorithms:
    x = [teg.state_durations[k] * k for k in range(teg.K)]
    y = [capacity for capacity in detla_capacities_by_algo[algorithm][1:]]
    
    plt.plot(x, y, label=display_name, linewidth=2.5)


# baseline
# num_gs * duration * deep space data rate
ogs_bit_rate_gigabit = 50 / 1000
num_ogs = 2 # because only two are generally visable at a time
dte_cap_x = [teg.state_durations[k] * k for k in range(teg.K)]
dte_cap_y = [ogs_bit_rate_gigabit * teg.state_durations[k] * num_ogs for k in range(teg.K)]
plt.plot(dte_cap_x, dte_cap_y, label="DTE Capacity", linewidth=2.5, color="gold", linestyle="dashed")

# Labels and formatting
plt.ylabel("Capacity [Gigabits]")
plt.ylim(0, 70)
plt.yticks(np.arange(0, 70, 10))

seconds_per_hour = 3600.0
ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(24 * seconds_per_hour, ax.get_xlim()[1] + 24 * seconds_per_hour, 24 * seconds_per_hour)))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Day {int(x // seconds_per_hour / 24)}"))
ax.set_xlabel("Time [Days]")

xmin, xmax = ax.get_xlim()
pad = 2 * 3600
ax.set_xlim(xmin - pad, xmax + pad)

plt.grid(linestyle='-', color='0.95')
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=5,
    frameon=False
)

# Save the figure
file_name = "dte_capacity_timeline"
plt.savefig(os.path.join("analysis", "mars_earth_dte_realistic_scenario_analysis", f"{file_name}.pdf"), format="pdf", bbox_inches="tight")
plt.savefig(os.path.join("analysis", "mars_earth_dte_realistic_scenario_analysis", f"{file_name}.png"), format="png", bbox_inches="tight", dpi=300)
