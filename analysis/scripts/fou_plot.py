import os
import pickle
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
from laser_link_scheduler.models.link_acq_delay import (
    link_acq_delay_ipn_fou,
    link_acq_delay_leo_fou,
)


plt.rcParams.update({"font.size": 22})
plt.rc("legend", fontsize=16)
plt.rcParams.update({"font.family": "Times New Roman"})

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

ipn_acq = []
leo_acq = []

fou_r = np.arange(0.25, 2.0 + 0.0001, 0.25)

for fou in fou_r:
    ipn_acq.append(link_acq_delay_ipn_fou(fou))
    leo_acq.append(link_acq_delay_leo_fou(fou))

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(fou_r, leo_acq, label="LEO Acquisition Delay", linewidth=2.5)
plt.plot(fou_r, ipn_acq, label="IPN Acquisition Delay", linewidth=2.5)

plt.ylabel("Acquisition Delay [sec]")
plt.xlabel("Field of Uncertainty [deg]")
plt.legend()

plt.grid(linestyle="-", color="0.95")

y_min = 0
y_max = max(ipn_acq) + 10
y_step = 100

plt.ylim(y_min, y_max + 10)
ax.set_yticks([y_min] + list(np.arange(y_step, y_max + 0.01, y_step)))

ax.set_xticks(fou_r)

bbox = dict(boxstyle="round", fc="0.9")
arrowprops = dict(
    arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=10"
)

file_name = "acq_delay_by_fou"
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

plt.show()
