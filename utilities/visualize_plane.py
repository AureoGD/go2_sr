import numpy as np
import matplotlib.pyplot as plt

# ---- sample contacts ----
contacts = np.array([[0.113, 0.335, 0.044], [0.204, 0.165, 0.039], [-0.269, 0.321, 0.048], [-0.183, 0.167, 0.04]],
                    dtype=np.float32)

p1, p2, p3, p4 = contacts

# ---- plane normal (unit) ----
n = np.array([0.007, -0.036, 0.999], dtype=np.float32)
n /= np.linalg.norm(n)

# ---- geometry (R-method) ----
R = 0.25
d_rot = (p4 - p2)
d_rot /= np.linalg.norm(d_rot)
d_perp = np.cross(n, d_rot)
d_perp /= np.linalg.norm(d_perp)

if (np.dot(d_perp, p1 - p2) < 0) and (np.dot(d_perp, p3 - p2) < 0):
    d_perp = -d_perp

Pr1_world = p2 + R * d_perp
Pr2_world = p4 + R * d_perp

# ---- NEW FOOT INITIAL POSITION (example) ----
# in real use, this will come from state: foot_init = robot.data.oMf[foot_frame].translation
foot_init = np.array([-0.154, -0.104, 0.176])  # placeholder

# ---- NEW Point lying ON rotation line, distance R from p4 ----
P3_line = p4 + d_rot * R  # base xy direction
# matching z to the foot's current altitude
P3_line[2] = (foot_init[2] + Pr2_world[2]) / 2

# ---- plot ----
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


def draw():
    ax.clear()

    plane_poly = np.array([p1, p2, p4, p3])
    ax.plot_trisurf(plane_poly[:, 0], plane_poly[:, 1], plane_poly[:, 2], alpha=0.25)

    for a, b in [(p1, p2), (p2, p4), (p4, p3), (p3, p1)]:
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color='gray')

    ax.plot([p2[0], p4[0]], [p2[1], p4[1]], [p2[2], p4[2]], color='orange', linewidth=3, label="Support line P2–P4")

    # reference perpendicular points
    ax.scatter(Pr1_world[0], Pr1_world[1], Pr1_world[2], color='red', s=100)
    ax.text(Pr1_world[0], Pr1_world[1], Pr1_world[2], " Pr1", color='red')

    ax.scatter(Pr2_world[0], Pr2_world[1], Pr2_world[2], color='red', s=100)
    ax.text(Pr2_world[0], Pr2_world[1], Pr2_world[2], " Pr2", color='red')

    ax.plot([Pr1_world[0], Pr2_world[0]], [Pr1_world[1], Pr2_world[1]], [Pr1_world[2], Pr2_world[2]],
            '--r',
            linewidth=2,
            label="Reference rail")

    # NEW P3-line point
    ax.scatter(P3_line[0], P3_line[1], P3_line[2], color='blue', s=120)
    ax.text(P3_line[0], P3_line[1], P3_line[2], " P3_line", color='blue')

    # foot initial pos
    ax.scatter(foot_init[0], foot_init[1], foot_init[2], color='green', s=80)
    ax.text(foot_init[0], foot_init[1], foot_init[2], " foot_init", color='green')

    for pt, name in zip([p1, p2, p3, p4], ["P1", "P2", "P3", "P4"]):
        ax.scatter(pt[0], pt[1], pt[2], color='black', s=40)
        ax.text(pt[0], pt[1], pt[2], f" {name}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim(0.0, 0.5)
    ax.set_zticks([0.0, 0.25, 0.5])
    ax.legend()


draw()
ax.set_title("Updated Reference Visualization — Pr1, Pr2 and Rotation-line P3 (R method)")


# keyboard shortcuts
def on_key(event):
    if event.key == 't':
        ax.view_init(elev=90, azim=-90)
        ax.set_title("Top View")
    elif event.key == 's':
        ax.view_init(elev=0, azim=0)
        ax.set_title("Side View")
    elif event.key == 'f':
        ax.view_init(elev=0, azim=-90)
        ax.set_title("Front View")
    elif event.key == 'd':
        ax.view_init(elev=30, azim=-60)
        ax.set_title("3D Default")
    fig.canvas.draw_idle()


cid = fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

Pr1_world, Pr2_world, P3_line
