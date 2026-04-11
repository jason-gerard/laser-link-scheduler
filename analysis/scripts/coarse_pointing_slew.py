import os

from matplotlib.patches import Circle, FancyArrow
import matplotlib.pyplot as plt


# Starting and target az/el (degrees)
start_az, start_el = 0, 0
target_az, target_el = 45, 30

# FOU radius (degrees)
fou_radius = 5

# Current terminal boresight offset (a couple degrees off center)
current_boresight_az = start_az + 2
current_boresight_el = start_el + 1

# Compute deltas
delta_az = target_az - current_boresight_az
delta_el = target_el - current_boresight_el

fig, ax = plt.subplots(figsize=(8, 6))

# Draw starting terminal FOU
start_circle = Circle(
    (start_az, start_el),
    fou_radius,
    color="blue",
    alpha=0.2,
    label="Current link terminal FOU",
)
ax.add_patch(start_circle)

# Current terminal boresight (offset from center)
ax.scatter(
    current_boresight_az,
    current_boresight_el,
    c="blue",
    s=80,
    marker="o",
    label="Current terminal boresight",
)

# Draw target terminal FOU
target_circle = Circle(
    (target_az, target_el),
    fou_radius,
    color="red",
    alpha=0.2,
    label="Next link terminal FOU",
)
ax.add_patch(target_circle)

# Draw destination boresight
ax.scatter(
    target_az,
    target_el,
    c="red",
    s=80,
    marker="o",
    label="Pointing terminal boresight",
)

# Main slew path (dashed line)
ax.plot(
    [current_boresight_az, target_az],
    [current_boresight_el, target_el],
    linestyle=(0, (8, 6)),
    color="black",
    linewidth=1,
    label="Slew path",
)

# Arrowhead for slew path
ax.annotate(
    "",
    xy=(target_az, target_el),
    xycoords="data",
    xytext=(current_boresight_az, current_boresight_el),
    textcoords="data",
    arrowprops=dict(
        arrowstyle="->", color="black", linestyle=(0, (8, 6)), lw=1
    ),
)

# Azimuth component (green horizontal arrow)
az_arrow = FancyArrow(
    current_boresight_az,
    current_boresight_el,
    delta_az,
    0,
    width=0.1,
    length_includes_head=True,
    head_width=1.2,
    head_length=1.5,
    color="green",
    label="Azimuth component",
)
ax.add_patch(az_arrow)

# Elevation component (yellow vertical arrow, starting at boresight)
el_arrow = FancyArrow(
    current_boresight_az,
    current_boresight_el,
    0,
    delta_el,
    width=0.1,
    length_includes_head=True,
    head_width=1.2,
    head_length=1.5,
    color="gold",
    label="Elevation component",
)
ax.add_patch(el_arrow)

# Annotate delta values
ax.text(
    current_boresight_az + delta_az / 2,
    current_boresight_el - 2,
    f"Δaz = {delta_az:.1f}°",
    color="green",
    ha="center",
    fontsize=10,
)

ax.text(
    current_boresight_az + 2,
    current_boresight_el + delta_el / 2,
    f"Δel = {delta_el:.1f}°",
    color="goldenrod",
    va="center",
    fontsize=10,
)

# Labels and formatting
ax.set_xlabel("Azimuth (degrees)")
ax.set_ylabel("Elevation (degrees)")
ax.set_title("Coarse Pointing / Slew Phase of FSO Terminal Alignment")

# Legend below the plot with multiple columns
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    fancybox=True,
    shadow=False,
    ncol=3,
)

ax.grid(True, linestyle="--", alpha=0.6)

# Axis limits
ax.set_xlim(-10, 60)
ax.set_ylim(-10, 40)

file_name = "coarse_pointing_slew"
plt.savefig(
    os.path.join("analysis", f"{file_name}.png"),
    format="png",
    bbox_inches="tight",
    dpi=300,
)

plt.show()
