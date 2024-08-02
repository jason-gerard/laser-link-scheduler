import json

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
from matplotlib.colors import ListedColormap

with open("./analysis/metrics.json", "r") as f:
    metrics = json.load(f)
    
W_avg = metrics["W_avg"]

max_val = max(map(max, W_avg))

viridis = mpl.colormaps['viridis'].resampled(256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([1, 1, 1, 1])
newcolors[:1, :] = white
newcmp = ListedColormap(newcolors)

fig, ax = plt.subplots(1, 1)

psm = ax.pcolormesh(metrics["W_avg"], cmap=newcmp, vmax=max_val)

dim = np.arange(0, len(W_avg)+1)
ax.grid(True, which='both', axis='both', linestyle='-', color='grey')
ax.set_xticks(dim, minor=True)
ax.set_yticks(dim, minor=True)

ax.set_xlim(dim[0], dim[-1])
ax.set_ylim(dim[0], dim[-1])

fig.colorbar(psm, ax=ax)
plt.show()
