import os
import pickle
import pprint
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import gaussian_kde

import matplotlib as mpl
from matplotlib.colors import ListedColormap

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from time_expanded_graph import TimeExpandedGraph
from weights import compute_all_delays

plt.rcParams.update({"font.size": 26})
plt.rc("legend", fontsize=22)
plt.rcParams.update({"font.family": "Times New Roman"})

tegs = []

# report_id = 1748770993
report_id = 1739570900
# report_id = 1748302518
# report_id = 1748302577
file_names = [
    "lls_gs_mars_earth_scenario_inc_64.pkl",
    "lls_gs_mars_earth_scenario_inc_60.pkl",
]
# file_name = "lls_mip_gs_mars_earth_scenario_inc_reduced_4.pkl"
# file_name = "lls_gs_mars_earth_scenario_inc_reduced_4.pkl"
for file_name in file_names:
    with open(f"reports/{report_id}/{file_name}", "rb") as f:
        teg: TimeExpandedGraph = pickle.load(f)
        tegs.append(teg)

all_pointing_delays = []
all_pointing_delays_with_node = []
all_link_acq_delays = []

for teg in tegs:
    delays_by_node = {node: [] for node in teg.nodes}

    for k in range(teg.K):
        for tx_oi_idx in range(teg.N):
            for rx_oi_idx in range(teg.N):
                if teg.graphs[k][tx_oi_idx][rx_oi_idx] == 1:
                    res = compute_all_delays(
                        tx_oi_idx,
                        rx_oi_idx,
                        teg.graphs[:k],
                        teg.state_durations[k],
                        teg.pos,
                        teg.optical_interfaces_to_node,
                        teg.nodes,
                    )
                    if res is None:
                        continue
                    else:
                        link_acq_delay = res[0]
                        pd1, idx1, idx2, idx1_rx = res[1]
                        pd2, idx2, idx1, idx2_rx = res[2]
                        pd1 += 2
                        pd2 += 2

                    tx_node = teg.nodes[teg.optical_interfaces_to_node[tx_oi_idx]]
                    rx_node = teg.nodes[teg.optical_interfaces_to_node[rx_oi_idx]]

                    # state duration, pointing delay, link acq delay
                    delays_by_node[tx_node].append(
                        (teg.state_durations[k], pd1, link_acq_delay)
                    )
                    delays_by_node[rx_node].append(
                        (teg.state_durations[k], pd2, link_acq_delay)
                    )

                    all_pointing_delays.append(pd1)
                    all_pointing_delays.append(pd2)
                    if (
                        teg.nodes[idx1].startswith("2")
                        and teg.nodes[idx2].startswith("2")
                        and teg.nodes[idx1_rx].startswith("2")
                    ):
                        pass
                    else:
                        all_pointing_delays_with_node.append(
                            (teg.nodes[idx1], teg.nodes[idx2], teg.nodes[idx1_rx], pd1)
                        )
                    if (
                        teg.nodes[idx1].startswith("2")
                        and teg.nodes[idx2].startswith("2")
                        and teg.nodes[idx2_rx].startswith("2")
                    ):
                        pass
                    else:
                        all_pointing_delays_with_node.append(
                            (teg.nodes[idx2], teg.nodes[idx1], teg.nodes[idx2_rx], pd2)
                        )
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
        retargeting_duty_cycle[node] = total_eff_time / total_time
        # print(node, total_time, total_eff_time)

    # pprint.pprint(retargeting_duty_cycle)

    network_retargeting_duty_cycle = network_total_eff_time / network_total_time
    print("network retargeting duty cycle", network_retargeting_duty_cycle)

all_pointing_delays = np.array(all_pointing_delays)
all_link_acq_delays = np.array(all_link_acq_delays)

all_pointing_delays_with_node = sorted(
    all_pointing_delays_with_node, key=lambda x: x[-1]
)
# pprint.pprint(all_pointing_delays_with_node)

all_pointing_delays = all_pointing_delays[all_pointing_delays > 0]
all_link_acq_delays = all_link_acq_delays[all_link_acq_delays > 0]

kde_pointing = gaussian_kde(all_pointing_delays)
kde_acquisition = gaussian_kde(all_link_acq_delays)

# Create an x-axis range for each
x_pointing = np.linspace(0, all_pointing_delays.max() + 5, 200)
x_acquisition = np.linspace(0, all_link_acq_delays.max() + 20, 200)

pointing_bins = np.linspace(2, all_pointing_delays.max() + 5, 40)
acquisition_bins = np.linspace(2, all_link_acq_delays.max() + 20, 40)

plt.figure(figsize=(10, 6))

ax = plt.gca()
ax.set_axisbelow(True)

plt.hist(
    all_pointing_delays,
    bins=pointing_bins,
    density=True,
    alpha=0.4,
    color="blue",
    label="Pointing Delay Histogram",
)
plt.hist(
    all_link_acq_delays,
    bins=acquisition_bins,
    density=True,
    alpha=0.4,
    color="orange",
    label="Acquisition Delay Histogram",
)

plt.plot(x_pointing, kde_pointing(x_pointing), label="Pointing Delay PDF", color="blue")
plt.plot(
    x_acquisition,
    kde_acquisition(x_acquisition),
    label="Acquisition Delay PDF",
    color="orange",
)

plt.xlim(-0.25, 250)
plt.xticks(np.arange(0, 251, 25))

plt.ylim(0, 0.18)
plt.yticks(np.arange(0, 0.17, 0.02))

# plt.title('Probability Distribution of Pointing and Acquisition Delays')
plt.xlabel("Delay (seconds)")
plt.ylabel("Probability Density")
plt.grid(linestyle="-", color="0.95")
plt.legend()

bbox = dict(boxstyle="round", fc="0.9")
arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10")

plt.annotate(
    "IPN-to-IPN",
    fontsize=20,
    xy=(3.0, 0.135),
    xytext=(4.0, 0.16),
    textcoords="data",
    bbox=bbox,
    arrowprops=arrowprops,
    ha="left",
    va="bottom",
)
plt.annotate(
    "Ground/LEO-to-IPN",
    fontsize=20,
    xy=(37, 0.02),
    xytext=(50, 0.13),
    textcoords="data",
    bbox=bbox,
    arrowprops=arrowprops,
    ha="center",
    va="bottom",
)
plt.annotate(
    "LEO-to-LEO",
    fontsize=20,
    xy=(85, 0.015),
    xytext=(85, 0.05),
    textcoords="data",
    bbox=bbox,
    arrowprops=arrowprops,
    ha="center",
    va="bottom",
)

plt.annotate(
    "LEO Acq",
    fontsize=20,
    xy=(48.0, 0.085),
    xytext=(46.0, 0.1),
    textcoords="data",
    bbox=bbox,
    arrowprops=arrowprops,
    ha="left",
    va="bottom",
)

plt.annotate(
    "IPN Acq",
    fontsize=20,
    xy=(210.0, 0.012),
    xytext=(210.0, 0.04),
    textcoords="data",
    bbox=bbox,
    arrowprops=arrowprops,
    ha="center",
    va="bottom",
)

plt.tight_layout()

file_name = "retargeting delay pdf".replace(" ", "_").replace("/", "_")
plt.savefig(
    os.path.join("analysis", f"{file_name}.pdf"), format="pdf", bbox_inches="tight"
)
plt.savefig(
    os.path.join("analysis", f"{file_name}.png"),
    format="png",
    bbox_inches="tight",
    dpi=300,
)

plt.show()
