import numpy as np
import matplotlib.pyplot as plt

data = np.load("tpe_simulation_log.npz")

alpha = data["alpha"]
omega = data["omega_x"]
v_abs = data["v_abs"]
controller = data["controller"]
phase = data["tpe_phase"]

t = np.arange(len(alpha))

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

axs[0].plot(t, alpha)
axs[0].set_ylabel("alpha")

axs[1].plot(t, omega)
axs[1].set_ylabel("omega_x")

axs[2].plot(t, controller)
axs[2].set_ylabel("controller")

axs[3].plot(t, phase)
axs[3].set_ylabel("TPE phase")

axs[3].set_xlabel("time step")

plt.tight_layout()
plt.show()
