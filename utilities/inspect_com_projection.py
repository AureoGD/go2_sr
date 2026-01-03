import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# -----------------------------
# Given points (XY projection)
# -----------------------------

contacts = np.array([[-0.189, -0.032, 0.094], [0.17, 0.11, 0.12], [-0.188, -0.335, 0.047], [0.031, 0.221, 0.221],
                     [-0.313, 0.057, 0.212]])

rr_pivot = contacts[0, 0:2].flatten()
fr_pivot = contacts[1, 0:2].flatten()
rl_foot = contacts[2, 0:2].flatten()
fr_foot = contacts[3, 0:2].flatten()
rr_foot = contacts[4, 0:2].flatten()

com = np.array([0.006, -0.002])

points = np.array([fr_foot, rr_foot, rl_foot, fr_pivot])

# -----------------------------
# Convex hull (support polygon)
# -----------------------------
hull = ConvexHull(points)
poly = points[hull.vertices]

# -----------------------------
# Half-space Ax <= b
# -----------------------------
A, b = [], []
for i in range(len(poly)):
    p1 = poly[i]
    p2 = poly[(i + 1) % len(poly)]
    edge = p2 - p1
    normal = np.array([edge[1], -edge[0]])
    normal = normal / np.linalg.norm(normal)
    A.append(normal)
    b.append(normal @ p1)

A = np.array(A)
b = np.array(b)


# -----------------------------
# Chebyshev center optimization
# -----------------------------
def obj(x):
    return -x[2]  # maximize radius


def cons_fun(x):
    cx, cy, r = x
    return b - A @ np.array([cx, cy]) - r


x0 = np.array([np.mean(points[:, 0]), np.mean(points[:, 1]), 0.05])
cons = [{'type': 'ineq', 'fun': lambda x, i=i: cons_fun(x)[i]} for i in range(len(b))]
res = minimize(obj, x0, constraints=cons)

cx, cy, r = res.x

# -----------------------------
# Hexagon approximation
# -----------------------------
angles = np.linspace(0, 2 * np.pi, 7)[:-1]
hexagon = np.c_[cx + r * np.cos(angles), cy + r * np.sin(angles)]

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6, 6))

# Support polygon
plt.fill(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]], alpha=0.25, label="Support polygon")

# Chebyshev circle
circle = plt.Circle((cx, cy), r, fill=False, linewidth=2, label="Chebyshev circle")
plt.gca().add_patch(circle)

# Hexagon approximation
plt.plot(np.r_[hexagon[:, 0], hexagon[0, 0]],
         np.r_[hexagon[:, 1], hexagon[0, 1]],
         linestyle="--",
         linewidth=2,
         label="Hexagon approximation")

# Pivot line (FR pivot → CoM projection)
plt.plot([fr_pivot[0], rr_pivot[0]], [fr_pivot[1], rr_pivot[1]], linewidth=2, label="Pivot line (FR → CoM)")

# Points
plt.scatter(points[:, 0], points[:, 1], zorder=5, label="Contacts / Pivot")
plt.scatter(rr_pivot[0], rr_pivot[1], zorder=5, label="Contacts / Pivot")
plt.scatter(cx, cy, marker='x', s=80, label="Chebyshev center")
plt.scatter(com[0], com[1], marker='*', s=120, label="CoM")

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Support polygon, Chebyshev circle, hexagon & pivot line")
plt.show()

(cx, cy, r)
