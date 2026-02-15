import os
import csv
import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ----------------------------
# Plot style (top-of-file)
# ----------------------------
plt.rcParams.update({'font.size': 22})
plt.rc('legend', fontsize=16)
plt.rcParams.update({'font.family': 'Times New Roman'})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'pdf.fonttype': 42})

# ----------------------------
# Load CSV (your starting point)
# ----------------------------
report_id = 1770422179
file_name = "lls_sensitivity_analysis_alpha"
path = os.path.join("reports", str(report_id), f"{report_id}_report.csv")

with open(path, "r", newline="") as f:
    report = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]


# ----------------------------
# Helpers
# ----------------------------
def to_float(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


_alpha_re = re.compile(r"alpha[_=]([0-9]*\.?[0-9]+)")
_inc_re = re.compile(r"inc_(\d+)$")

def parse_alpha(algorithm_name):
    if algorithm_name is None:
        return np.nan
    s = str(algorithm_name).strip()
    m = _alpha_re.search(s)
    if not m:
        return np.nan
    try:
        return float(m.group(1))
    except ValueError:
        return np.nan


# ----------------------------
# Extract data for plot
# ----------------------------
xs = []
ys = []
cs = []

for r in report:
    cap_mb = to_float(r.get("Capacity"))
    fair = to_float(r.get("Jain's fairness index"))
    alpha = parse_alpha(r.get("Algorithm"))

    scenario = (r.get("Scenario") or "").strip()
    m = _inc_re.search(scenario)
    inc = int(m.group(1)) if m else None
    # if inc == 4:
    #     continue

    if np.isfinite(cap_mb) and np.isfinite(fair) and np.isfinite(alpha):
        cap = cap_mb / 1000.0 / 1000.0   # MBits -> TBits
        xs.append(cap)
        ys.append(fair)
        cs.append(alpha)

xs = np.asarray(xs, dtype=float)
ys = np.asarray(ys, dtype=float)
cs = np.asarray(cs, dtype=float)

if xs.size == 0:
    raise RuntimeError("No valid points found. Check column names and alpha parsing.")


# ----------------------------
# Axis limits (computed from the data)
# ----------------------------
x_min, x_max = (4.0, xs.max())
x_step = 1
y_min, y_max = float(ys.min()), float(ys.max())

x_lim = (3.5, 7)
y_lim = (0.0, 1.0)


# ----------------------------
# Scatter plot (color is alpha in [0, 1])
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 6))

sc = ax.scatter(
    xs,
    ys,
    c=cs,
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    s=40,
    alpha=0.9,
)

ax.set_xlabel("Capacity [Terabits/day]")
ax.set_ylabel("Jain's fairness index")

ax.set_xlim(*x_lim)
ax.set_ylim(*y_lim)
ax.set_xticks(np.arange(4.0, int(x_max+2), x_step).tolist())

ax.grid(True, which="both", linewidth=0.6, alpha=0.35)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r"$\alpha$", rotation=0, labelpad=18)

plt.tight_layout()
# plt.show()

plt.savefig(
    os.path.join("analysis", "sensitivity_analysis", f"{file_name}.pdf"),
    format="pdf",
    bbox_inches="tight"
)
plt.savefig(
    os.path.join("analysis", "sensitivity_analysis", f"{file_name}.png"),
    format="png",
    bbox_inches="tight",
    dpi=300,
)
