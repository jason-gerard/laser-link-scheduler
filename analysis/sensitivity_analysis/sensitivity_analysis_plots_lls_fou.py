import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, FuncFormatter


# ----------------------------
# Configuration
# ----------------------------
REPORT_ID = 1785103892
FILE_NAME = "lls_sensitivity_analysis_fou"

REPORT_PATH = Path("reports") / str(REPORT_ID) / f"{REPORT_ID}_report.csv"
OUTPUT_DIR = Path("analysis") / "sensitivity_analysis"

FOU_MIN = 2.5
FOU_MAX = 30.0

CAPACITY_TICK_STEP = 1.0
CAPACITY_AXIS_PADDING = 0.75

FAIRNESS_MIN = 0.7
FAIRNESS_MAX = 1.0
FAIRNESS_TICK_STEP = 0.1

FOU_PATTERN = re.compile(r"fou[_=]([0-9]*\.?[0-9]+)")


# ----------------------------
# Plot style
# ----------------------------
plt.rcParams.update(
    {
        "font.size": 22,
        "font.family": "Times New Roman",
        "legend.fontsize": 16,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ----------------------------
# Helper functions
# ----------------------------
def to_float(value):
    """Convert a CSV value to float, returning NaN when invalid."""
    if value is None:
        return np.nan

    value = str(value).strip().replace(",", "")

    if not value:
        return np.nan

    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_fou(algorithm_name):
    """Extract the FOU value from the algorithm name."""
    if algorithm_name is None:
        return np.nan

    match = FOU_PATTERN.search(str(algorithm_name).strip())

    if match is None:
        return np.nan

    return to_float(match.group(1))


def format_capacity_tick(value, _position):
    """
    Display whole-number capacity ticks without a decimal and half-step
    ticks with one decimal.

    Examples:
        10.0 -> 10
        12.5 -> 12.5
        15.0 -> 15
    """
    if np.isclose(value, round(value)):
        return f"{int(round(value))}"

    return f"{value:.1f}"


# ----------------------------
# Load and extract data
# ----------------------------
capacities = []
fairness_values = []
fou_values = []

with REPORT_PATH.open("r", newline="") as file:
    reader = csv.DictReader(file, skipinitialspace=True)

    for row in reader:
        capacity_mbits = to_float(row.get("Capacity"))
        fairness = to_float(row.get("Jain's fairness index"))
        fou = parse_fou(row.get("Algorithm"))

        if not np.all(np.isfinite([capacity_mbits, fairness, fou])):
            continue

        # Convert the reported capacity to Tbits/day.
        capacity_tbits = capacity_mbits / 1_000_000.0 / 100.0

        capacities.append(capacity_tbits)
        fairness_values.append(fairness)
        fou_values.append(fou)

capacities = np.asarray(capacities, dtype=float)
fairness_values = np.asarray(fairness_values, dtype=float)
fou_values = np.asarray(fou_values, dtype=float)

if capacities.size == 0:
    raise RuntimeError(
        "No valid data points were found. Check the CSV column names "
        "and the FOU values in the Algorithm column."
    )


# ----------------------------
# Calculate axis limits and ticks
# ----------------------------
capacity_tick_min = (
    np.floor(capacities.min() / CAPACITY_TICK_STEP)
    * CAPACITY_TICK_STEP
)

capacity_tick_max = (
    np.ceil(capacities.max() / CAPACITY_TICK_STEP)
    * CAPACITY_TICK_STEP
)

if np.isclose(capacity_tick_min, capacity_tick_max):
    capacity_tick_max += CAPACITY_TICK_STEP

capacity_ticks = np.arange(
    capacity_tick_min,
    capacity_tick_max + CAPACITY_TICK_STEP / 2,
    CAPACITY_TICK_STEP,
)

fairness_ticks = np.arange(
    FAIRNESS_MIN,
    FAIRNESS_MAX + FAIRNESS_TICK_STEP / 2,
    FAIRNESS_TICK_STEP,
)

# Add space before the first x tick so it does not collide with the
# first y tick label. The displayed tick values remain unchanged.
capacity_axis_min = capacity_tick_min - CAPACITY_AXIS_PADDING
capacity_axis_max = capacity_tick_max


# ----------------------------
# Create plot
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 6))

scatter = ax.scatter(
    capacities,
    fairness_values,
    c=fou_values,
    cmap="viridis",
    vmin=FOU_MIN,
    vmax=FOU_MAX,
    s=40,
    alpha=0.9,
)

ax.set_xlabel("Capacity [Terabits/day]")
ax.set_ylabel("Jain's fairness index")

ax.set_xlim(capacity_axis_min, capacity_axis_max)
ax.set_xticks(capacity_ticks)
ax.xaxis.set_major_formatter(FuncFormatter(format_capacity_tick))

ax.set_ylim(FAIRNESS_MIN, FAIRNESS_MAX)
ax.set_yticks(fairness_ticks)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

ax.grid(
    True,
    which="major",
    linewidth=0.6,
    alpha=0.35,
)

colorbar = fig.colorbar(
    scatter,
    ax=ax,
    orientation="horizontal",
    pad=0.22,
)

colorbar.set_label("FOU half-angle (mrad)", labelpad=8)

fig.tight_layout()


# ----------------------------
# Save plot
# ----------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig.savefig(
    OUTPUT_DIR / f"{FILE_NAME}.pdf",
    format="pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / f"{FILE_NAME}.png",
    format="png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)