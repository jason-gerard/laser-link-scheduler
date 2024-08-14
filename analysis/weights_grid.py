import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import matplotlib as mpl
from matplotlib.colors import ListedColormap

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from time_expanded_graph import TimeExpandedGraph

# report_id = 1722737768
report_id = 1722740313
file_name = "lls_mars_earth_simple_scenario.pkl"
with open(f"reports/{report_id}/{file_name}", "rb") as f:
    teg: TimeExpandedGraph = pickle.load(f)
    
W_avg = np.sum(teg.W, axis=0) / teg.K

max_val = max(map(max, W_avg))

viridis = mpl.colormaps['viridis'].resampled(256)
newcolors = viridis(np.linspace(0, 1, int(max_val)))
white = np.array([1, 1, 1, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

fig, ax = plt.subplots(1, 1)

psm = ax.pcolormesh(W_avg, cmap=newcmp, vmax=max_val)

dim = np.arange(0, teg.N+1)
ax.grid(True, which='both', axis='both', linestyle='-', color='grey')

ax.set_xticks(dim)
ax.set_yticks(dim)

original_labels = [str(label) for label in ax.get_xticks()]
labels_of_interest = [str(i) for i in [0, 4, 20, 24]]
new_labels = [label if label in labels_of_interest else "" for label in original_labels]
ax.set_xticklabels(new_labels)

original_labels = [str(label) for label in ax.get_yticks()]
labels_of_interest = [str(i) for i in [0, 4, 20, 24]]
new_labels = [label if label in labels_of_interest else "" for label in original_labels]
ax.set_yticklabels(new_labels)

ax.yaxis.set_label_text("Transmitting Satellite ID")

# Second X-axis
ax2 = ax.twiny()

ax2.spines["bottom"].set_position(("axes", -0.075))
ax2.tick_params('both', length=0, width=0, which='minor')
ax2.tick_params('both', direction='in', which='major')
ax2.xaxis.set_ticks_position("bottom")
ax2.xaxis.set_label_position("bottom")

ax2.set_xticks([0, 4, 20, 24])
ax2.xaxis.set_major_formatter(ticker.NullFormatter())
ax2.xaxis.set_minor_locator(ticker.FixedLocator([2, 12, 22]))
ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(['Earth Relay', 'Mars Orbiter', 'Mars Relay']))

ax.set_xlim(dim[0], dim[-1])
ax.set_ylim(dim[0], dim[-1])

plt.xlabel("Receiving Satellite ID")
# plt.ylabel("Transmitting Satellite ID")

cbar = fig.colorbar(psm, ax=ax)
cbar.set_label('Average Delta Capacity & DCT')

plt.savefig(
    os.path.join("analysis", "weights_grid.pdf"),
    format="pdf",
    bbox_inches='tight'
)
plt.savefig(
    os.path.join("analysis", "weights_grid.png"),
    format="png",
    bbox_inches='tight'
)
plt.show()
