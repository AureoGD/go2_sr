import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('/home/CCT/9791086/mujoco_sr/data.csv')

data.columns = data.columns.str.strip()

print(data.columns)
# Force numeric conversion
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna().reset_index(drop=True)

modes = data['mode'].values
timesteps = np.arange(len(data))

# Background colors per mode
mode_colors = {0: '#f0f0f0', 1: '#d9edf7', 2: '#dff0d8', 3: '#fcf8e3', 4: '#f2dede', 5: '#e8daef', 6: '#f5cba7'}


def add_mode_background(ax, modes):
    start = 0
    current_mode = modes[0]

    for i in range(1, len(modes)):
        if modes[i] != current_mode:
            ax.axvspan(start, i, alpha=0.3, color=mode_colors.get(int(current_mode), '#ffffff'))
            start = i
            current_mode = modes[i]

    ax.axvspan(start, len(modes), alpha=0.3, color=mode_colors.get(int(current_mode), '#ffffff'))


# ===== Create subplots =====
fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True)

# ---- Plot 1: intent_norm ----
add_mode_background(axes[0], modes)
axes[0].plot(timesteps, data['intent_norm'].values)
axes[0].set_ylabel("intent_norm")
axes[0].set_title("Intent Norm")

# ---- Plot 2: solver_motion ----
add_mode_background(axes[1], modes)
axes[1].plot(timesteps, data['solver_motion'].values)
axes[1].set_ylabel("solver_motion")
axes[1].set_title("Solver Motion")

# ---- Plot 3: actual_motion ----
add_mode_background(axes[2], modes)
axes[2].plot(timesteps, data['actual_motion'].values)
axes[2].set_ylabel("actual_motion (||delta_q||)")
axes[2].set_xlabel("Time step")
axes[2].set_title("Actual Motion")

add_mode_background(axes[3], modes)
lambda_scaled = np.tanh(np.log1p(data['lambda_max'].values))
axes[3].plot(timesteps, lambda_scaled)
axes[3].set_ylabel("log1p lambda_max")
axes[3].set_xlabel("Time step")
axes[3].set_title("log1p lambada_max")

add_mode_background(axes[4], modes)
primal_scaled = np.tanh(data['primal'].values / 0.5)
axes[4].plot(timesteps, primal_scaled)
axes[4].set_ylabel("primal")
axes[4].set_xlabel("Time step")
axes[4].set_title("Primal Residual")

dual_scaled = np.tanh(data['dual'].values / 0.5)
add_mode_background(axes[5], modes)
axes[5].plot(timesteps, dual_scaled)
axes[5].set_ylabel("dual")
axes[5].set_xlabel("Time step")
axes[5].set_title("Dual Residual")

plt.tight_layout()
plt.show()
