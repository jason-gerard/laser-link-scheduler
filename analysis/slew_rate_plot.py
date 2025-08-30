import os
import math
import pickle
import pprint
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import gaussian_kde

import matplotlib as mpl
from matplotlib.colors import ListedColormap
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from time_expanded_graph import TimeExpandedGraph
from weights import compute_all_delays

plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=16)
plt.rcParams.update({'font.family': 'Times New Roman'})

tegs = []

# report_id = 1748770993
report_id = 1739570900
# report_id = 1748302518
# report_id = 1748302577
file_names = ["lls_gs_mars_earth_scenario_inc_64.pkl"]
# file_name = "lls_mip_gs_mars_earth_scenario_inc_reduced_4.pkl"
# file_name = "lls_gs_mars_earth_scenario_inc_reduced_4.pkl"
for file_name in file_names:
    with open(f"reports/{report_id}/{file_name}", "rb") as f:
        teg: TimeExpandedGraph = pickle.load(f)
        tegs.append(teg)

all_pointing_delays = []
all_pointing_delays_gs = []

slew_rate_mult = 0.017453
mults = 9

for teg in tegs:
    for k in range(teg.K):
        for tx_oi_idx in range(teg.N):
            for rx_oi_idx in range(teg.N):
                if teg.graphs[k][tx_oi_idx][rx_oi_idx] == 1:
                    for i in range(1, mults):
                        res = compute_all_delays(
                            tx_oi_idx,
                            rx_oi_idx,
                            teg.graphs[:k],
                            teg.state_durations[k],
                            teg.pos,
                            teg.optical_interfaces_to_node,
                            teg.nodes,
                            slew_rate=i * slew_rate_mult
                        )
                        if res is None:
                            continue
                        else:
                            link_acq_delay = res[0]
                            pd1, idx1, idx2, idx1_rx = res[1]
                            pd2, idx2, idx1, idx2_rx = res[2]
                        
                        if teg.nodes[idx1].startswith("2") and teg.nodes[idx2].startswith("2") and teg.nodes[idx1_rx].startswith("2"):
                            all_pointing_delays.append((i, pd1))
                            all_pointing_delays.append((i, pd2))
                        elif teg.nodes[idx1].startswith("2") and teg.nodes[idx2].startswith("2") and teg.nodes[idx2_rx].startswith("2"):
                            all_pointing_delays.append((i, pd1))
                            all_pointing_delays.append((i, pd2))
                        else:
                            all_pointing_delays_gs.append((i, pd1+2))
                            all_pointing_delays_gs.append((i, pd2+2))
                    
                        # tx_node = teg.nodes[teg.optical_interfaces_to_node[tx_oi_idx]]
                        # rx_node = teg.nodes[teg.optical_interfaces_to_node[rx_oi_idx]]

# all_pointing_delays.sort()
# pprint.pprint(all_pointing_delays)

all_pointing_delays = np.array(all_pointing_delays)
all_pointing_delays_gs = np.array(all_pointing_delays_gs)

# Collect values per bucket
buckets = defaultdict(list)
for bucket_id, value in all_pointing_delays:
    buckets[bucket_id].append(value)

buckets_gs = defaultdict(list)
for bucket_id, value in all_pointing_delays_gs:
    buckets_gs[bucket_id].append(value)

# Compute averages per bucket, sorted by bucket_id
average_values = []
for bucket_id in sorted(buckets.keys()):
    avg = sum(buckets[bucket_id]) / len(buckets[bucket_id])
    average_values.append(avg)

print(average_values)

average_values_gs = []
for bucket_id in sorted(buckets_gs.keys()):
    avg = sum(buckets_gs[bucket_id]) / len(buckets_gs[bucket_id])
    average_values_gs.append(avg)

fig = plt.figure()
ax = fig.add_subplot(111)

x = sorted(buckets.keys())
y = average_values

plt.plot(x, y, label="Average Pointing Delay (LEO-to-LEO)", linewidth=2.5)

y_gs = average_values_gs

plt.plot(x, y_gs, label="Average Pointing Delay (IPN-to-IPN)", linewidth=2.5)

plt.ylabel("Average Pointing Delay [sec]")
plt.xlabel("Slew Rate [deg/s]")
plt.legend()

plt.grid(linestyle='-', color='0.95')

y_min = 0
# y_max = max(max(y), max(y_gs)) + 5
y_max = max(y) + 3
y_step = 10

plt.ylim(y_min, y_max)
ax.set_yticks([y_min] + list(np.arange(y_step, y_max + 0.01, y_step)))

ax.set_xticks(range(1, 9))
plt.xlim(0.75,8.25)

bbox = dict(boxstyle="round", fc="0.9")
arrowprops = dict(
    arrowstyle="->",
    connectionstyle="angle,angleA=0,angleB=90,rad=10")

# plt.annotate("Stepper motor", fontsize=13, xy=(2.0, 31),
#             xytext=(1.75, 45), textcoords='data',
#             bbox=bbox, arrowprops=arrowprops, ha='left', va='bottom')
# plt.annotate("Brushless DC\nw/ encoder", fontsize=13, xy=(8, 8),
#             xytext=(8, 20), textcoords='data',
#             bbox=bbox, arrowprops=arrowprops, ha='right', va='bottom')

plt.axvline(
    x=2.0,
    color='black',
    linestyle='--',
    linewidth=1,
    zorder=0,
)
plt.text(
    2.0,
    y_max - 20,
    "Stepper motor",
    fontsize=18,
    ha='left',
    va='top',
    bbox=bbox
)

# Brushless DC marker
plt.axvline(
    x=8.0,
    color='black',
    linestyle='--',
    linewidth=1,
    zorder=0,
)
plt.text(
    8.0,
    y_max-35,
    "Brushless DC\nw/ encoder",
    fontsize=18,
    ha='right',
    va='top',
    bbox=bbox
)

file_name = "average_pointing_delay_by_slew_rate"
plt.savefig(
    os.path.join("analysis", f"{file_name}.pdf"),
    format="pdf",
    bbox_inches="tight"
)
plt.savefig(
    os.path.join("analysis", f"{file_name}.png"),
    format="png",
    bbox_inches="tight",
    dpi=300,
)

plt.show()
