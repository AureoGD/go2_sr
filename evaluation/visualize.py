import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IC_FILE = Path(__file__).resolve().parent / "ics" / "ics_v1.json"

with open(IC_FILE) as f:
    data = json.load(f)

yaws = np.array([ic["yaw"] for ic in data["ics"]])
bins = np.array([ic["yaw_bin"] for ic in data["ics"]])
edges = np.array(data["metadata"]["bin_edges_rad"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# scatter: bin index vs yaw, edges as horizontal lines
ax1.scatter(bins, yaws, s=10, alpha=0.6)
for e in edges:
    ax1.axhline(e, color="gray", lw=0.5, alpha=0.5)
ax1.set_xlabel("yaw bin")
ax1.set_ylabel("yaw [rad]")
ax1.set_title("Samples per bin")

# histogram: overall coverage
ax2.hist(yaws, bins=edges, edgecolor="white")
ax2.set_xlabel("yaw [rad]")
ax2.set_ylabel("count")
ax2.set_title("Yaw distribution")

plt.tight_layout()
plt.show()
