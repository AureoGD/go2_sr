import os
import numpy as np
from go2_model import Go2ModelSim
import matplotlib.pyplot as plt  # Fixed import

go2 = Go2ModelSim(render=True)

T_SIM = 5
N_STEPS = int(T_SIM / go2.con_dt)

go2.qr = np.array([[0, 1.36, -2.65, 0, 1.36, -2.65, 0, 1.36, -2.65, 0, 1.36, -2.65]]).transpose()

state = []
eps_state = []
for _ in range(N_STEPS):
    go2.control_loop(mode=1)
    state.append(go2.robot_states.r_pos)
    eps_state.append(go2.robot_states.epsilon)
# Convert state list to numpy array for easier manipulation
state_array = np.array(state)
eps_array = np.array(eps_state)

# Create time vector based on control loop iterations
time = np.arange(0, len(state_array)) * go2.con_dt

# Plot the CoM position
plt.figure(figsize=(12, 8))

# Plot x, y, z positions separately
plt.subplot(3, 1, 1)
plt.plot(time, state_array[:, 0], 'b-', linewidth=2)
plt.ylabel('X Position (m)')
plt.title('Center of Mass Position')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, state_array[:, 1], 'r-', linewidth=2)
plt.ylabel('Y Position (m)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, state_array[:, 2], 'g-', linewidth=2)
plt.ylabel('Z Position (m)')
plt.xlabel('Time (s)')
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))

# Plot x, y, z positions separately
plt.subplot(4, 1, 1)
plt.plot(time, eps_array[:, 0], 'b-', linewidth=2)
plt.ylim([-1.05, 1.05])
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time, eps_array[:, 1], 'b-', linewidth=2)
plt.ylim([-1.05, 1.05])
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(time, eps_array[:, 2], 'b-', linewidth=2)
plt.ylim([-1.05, 1.05])
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time, eps_array[:, 3], 'b-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylim([-1.05, 1.05])
plt.grid(True)

plt.tight_layout()
plt.show()
