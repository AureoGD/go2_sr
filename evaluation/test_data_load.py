import numpy as np
from pathlib import Path

path = Path("evaluation/controller_data/rgc_mpc_data.npz")
data = np.load(path)

# --- see what's inside ---
for key in data.files:
    print(f"{key:30s} {data[key].shape} {data[key].dtype}")