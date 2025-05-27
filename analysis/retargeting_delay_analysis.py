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
from weights import compute_delays

plt.rcParams.update({'font.size': 18})
plt.rc('legend', fontsize=14)
plt.rcParams.update({'font.family': 'Times New Roman'})

tegs = []

report_id = 1748302473
# report_id = 1748302518
# report_id = 1748302577
file_name = "lls_gs_mars_earth_scenario_inc_24.pkl"
# file_name = "lls_mip_gs_mars_earth_scenario_inc_reduced_4.pkl"
# file_name = "lls_gs_mars_earth_scenario_inc_reduced_4.pkl"
with open(f"reports/{report_id}/{file_name}", "rb") as f:
    teg: TimeExpandedGraph = pickle.load(f)
    tegs.append(teg)

all_pointing_delays = []
all_link_acq_delays = []

for teg in tegs:
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
        retargeting_duty_cycle[node] = total_eff_time / total_time
        # print(node, total_time, total_eff_time)

    # pprint.pprint(retargeting_duty_cycle)

    network_retargeting_duty_cycle = network_total_eff_time / network_total_time
    print("network retargeting duty cycle", network_retargeting_duty_cycle)

all_pointing_delays = np.array(all_pointing_delays)
all_link_acq_delays = np.array(all_link_acq_delays)

all_pointing_delays = all_pointing_delays[all_pointing_delays > 0]
all_link_acq_delays = all_link_acq_delays[all_link_acq_delays > 0]

kde_pointing = gaussian_kde(all_pointing_delays)
kde_acquisition = gaussian_kde(all_link_acq_delays)

# Create an x-axis range for each
x_pointing = np.linspace(0, all_pointing_delays.max() + 5, 200)
x_acquisition = np.linspace(0, all_link_acq_delays.max() + 20, 200)

pointing_bins = np.linspace(0, all_pointing_delays.max() + 5, 40)
acquisition_bins = np.linspace(0, all_link_acq_delays.max() + 20, 40)

plt.figure(figsize=(10, 6))

plt.hist(all_pointing_delays, bins=pointing_bins, density=True, alpha=0.4, color='blue', label='Pointing Delay Histogram')
plt.hist(all_link_acq_delays, bins=acquisition_bins, density=True, alpha=0.4, color='orange', label='Acquisition Delay Histogram')

plt.plot(x_pointing, kde_pointing(x_pointing), label='Pointing Delay PDF', color='blue')
plt.plot(x_acquisition, kde_acquisition(x_acquisition), label='Acquisition Delay PDF', color='orange')

plt.title('Probability Distribution of Pointing and Acquisition Delays')
plt.xlabel('Delay (seconds)')
plt.ylabel('Probability Density')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
