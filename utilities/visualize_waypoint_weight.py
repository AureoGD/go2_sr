import numpy as np
import matplotlib.pyplot as plt

# Example values
N = 20  # horizon length
P1 = np.array([0.0, 0.0])  # first waypoint
P2 = np.array([1.0, 0.0])  # final target
x0 = np.array([0.0, -0.6])  # current foot location (example)
dist_total = np.linalg.norm(P2 - P1)

# simulate predicted positions x(k) along a curved guess path (just visual)
xs = np.linspace(x0, np.array([1.0, 0.0]), N)

# compute k0 dynamically at each predicted step
k_vals = np.arange(N)
sigmas = []
for k in k_vals:
    # distance to P1 for this hypothetical predicted state
    dist_now = np.linalg.norm(xs[k] - P1)
    k0 = N * (dist_now / dist_total)  # dynamic transition index
    # smooth sigmoid
    s = 0.6
    sigma = 1.0 / (1.0 + np.exp(-s * (k - k0)))
    sigmas.append(sigma)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(k_vals, sigmas, marker='o', label="σ(k) dynamic")
plt.title("Constraint-based Smooth Gating (Dynamic σ(k))")
plt.xlabel("Prediction Step k")
plt.ylabel("Weight toward P2")
plt.grid(True)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.show()
