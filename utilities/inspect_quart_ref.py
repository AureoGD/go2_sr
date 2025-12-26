# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# class RobotVisualizer:

#     def __init__(self):
#         pass

#     def rotation_matrix_to_quaternion(self, R):
#         R = np.asarray(R)
#         q = np.zeros(4)
#         trace = R[0, 0] + R[1, 1] + R[2, 2]
#         if trace > 0:
#             S = np.sqrt(trace + 1.0) * 2
#             q[3] = 0.25 * S
#             q[0] = (R[2, 1] - R[1, 2]) / S
#             q[1] = (R[0, 2] - R[2, 0]) / S
#             q[2] = (R[1, 0] - R[0, 1]) / S
#         elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
#             S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
#             q[3] = (R[2, 1] - R[1, 2]) / S
#             q[0] = 0.25 * S
#             q[1] = (R[0, 1] + R[1, 0]) / S
#             q[2] = (R[0, 2] + R[2, 0]) / S
#         elif R[1, 1] > R[2, 2]:
#             S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
#             q[3] = (R[0, 2] - R[2, 0]) / S
#             q[0] = (R[0, 1] + R[1, 0]) / S
#             q[1] = 0.25 * S
#             q[2] = (R[1, 2] + R[2, 1]) / S
#         else:
#             S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
#             q[3] = (R[1, 0] - R[0, 1]) / S
#             q[0] = (R[0, 2] + R[2, 0]) / S
#             q[1] = (R[1, 2] + R[2, 1]) / S
#             q[2] = 0.25 * S
#         return q / np.linalg.norm(q)

#     def quaternion_to_rotation_matrix(self, q):
#         x, y, z, w = q
#         return np.array([[1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
#                          [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
#                          [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2]])

#     def eps_reference(self, current_yaw=None, desired_yaw=None, plane_normal=None, pivot_direction=None):
#         if plane_normal is not None:
#             desired_z = plane_normal / np.linalg.norm(plane_normal)
#         else:
#             desired_z = np.array([0., 0., 1.])

#         if pivot_direction is not None:
#             desired_x = pivot_direction / np.linalg.norm(pivot_direction)
#         else:
#             if desired_yaw is not None:
#                 yaw = desired_yaw
#             elif current_yaw is not None:
#                 yaw = current_yaw
#             else:
#                 yaw = 0
#             desired_x = np.array([np.cos(yaw), np.sin(yaw), 0])

#         desired_y = np.cross(desired_z, desired_x)
#         if np.linalg.norm(desired_y) < 1e-6:
#             desired_y = np.array([0., 1., 0.])
#         else:
#             desired_y = desired_y / np.linalg.norm(desired_y)

#         desired_x = np.cross(desired_y, desired_z)
#         desired_x = desired_x / np.linalg.norm(desired_x)

#         R = np.column_stack([desired_x, desired_y, desired_z])
#         return self.rotation_matrix_to_quaternion(R), R

#     def get_best_fit_normal(self, points):
#         centroid = np.mean(points, axis=0)
#         centered = points - centroid
#         u, s, vh = np.linalg.svd(centered)
#         normal = vh[2, :]
#         if normal[2] < 0:
#             normal = -normal
#         return normal, centroid

# def plot_scenario(data):
#     viz = RobotVisualizer()

#     # --- Data Extraction ---
#     p_fr, p_rr = data['pc_fr'], data['pc_rr']
#     p_fl, p_rl = data['pc_fl'], data['pc_rl']
#     c_fr, c_rr = data['c_fr'], data['c_rr']
#     b_pos = data['b_pos'].flatten()
#     q_curr = data['epsilon'].flatten()

#     # --- Calculations ---
#     contact_points = np.array([p_fr, p_rr, c_fr, c_rr])
#     plane_normal, plane_centroid = viz.get_best_fit_normal(contact_points)

#     pivot_vec = c_fr - c_rr
#     pivot_dir = pivot_vec / np.linalg.norm(pivot_vec)

#     q_ref, R_ref = viz.eps_reference(plane_normal=plane_normal, pivot_direction=pivot_dir)
#     R_body = viz.quaternion_to_rotation_matrix(q_curr)

#     # --- Visualization ---
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')

#     # Green Ground Plane
#     mid_pt = b_pos
#     range_span = 1.0
#     xx, yy = np.meshgrid(np.linspace(mid_pt[0] - range_span, mid_pt[0] + range_span, 10),
#                          np.linspace(mid_pt[1] - range_span, mid_pt[1] + range_span, 10))
#     z_ground = plane_centroid[2] - (plane_normal[0] * (xx - plane_centroid[0]) + plane_normal[1] *
#                                     (yy - plane_centroid[1])) / plane_normal[2]
#     ax.plot_surface(xx, yy, z_ground, alpha=0.2, color='green', label='Best Fit Ground')

#     # Cyan Contact Surface
#     verts = [np.array([p_rr, p_fr, c_fr, c_rr])]
#     poly = Poly3DCollection(verts, alpha=0.6, facecolors='cyan', edgecolors='blue', linewidths=2)
#     ax.add_collection3d(poly)

#     center_poly = np.mean(verts[0], axis=0)
#     ax.text(center_poly[0], center_poly[1], center_poly[2], "  Contact Area", color='blue', fontweight='bold')

#     # Points
#     feet = np.array([p_fr, p_fl, p_rr, p_rl])
#     pivots = np.array([c_fr, c_rr])
#     ax.scatter(feet[:, 0], feet[:, 1], feet[:, 2], c='k', s=30, label='Feet')
#     ax.scatter(pivots[:, 0], pivots[:, 1], pivots[:, 2], c='orange', s=80, marker='^', label='Shoulders')
#     ax.scatter(*b_pos, c='r', s=100, marker='s', label='Base')

#     # Frames (Scale 0.1)
#     scale = 0.1
#     ax.quiver(*b_pos, *R_body[:, 0], color='r', length=scale, label='Body X')
#     ax.quiver(*b_pos, *R_body[:, 1], color='g', length=scale, label='Body Y')
#     ax.quiver(*b_pos, *R_body[:, 2], color='b', length=scale, label='Body Z')

#     ax.quiver(*center_poly, *R_ref[:, 0], color='r', linestyle='--', length=scale, label='Ref X')
#     ax.quiver(*center_poly, *R_ref[:, 1], color='g', linestyle='--', length=scale, label='Ref Y')
#     ax.quiver(*center_poly, *R_ref[:, 2], color='b', linestyle='--', length=scale, label='Ref Z')
#     ax.quiver(*center_poly, *plane_normal, color='k', length=scale * 1.5, linewidth=1, label='Best Fit Normal')

#     # Formatting
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Press "t" for Top View (XY), "s" for Side View')

#     # Auto Scale
#     all_points = np.vstack([feet, b_pos, pivots])
#     mid = np.mean(all_points, axis=0)
#     radius = 0.4
#     ax.set_xlim(mid[0] - radius, mid[0] + radius)
#     ax.set_ylim(mid[1] - radius, mid[1] + radius)
#     ax.set_zlim(mid[2] - radius, mid[2] + radius)

#     handles, labels = ax.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     ax.legend(by_label.values(), by_label.keys(), loc='upper left')

#     # --- KEYBOARD EVENT HANDLER ---
#     def on_key(event):
#         if event.key == 't':  # Top View
#             ax.view_init(elev=90, azim=-90)
#             ax.set_title('Top View (XY)')
#         elif event.key == 's':  # Side View (YZ plane roughly)
#             ax.view_init(elev=0, azim=0)
#             ax.set_title('Side View (XZ)')
#         elif event.key == 'f':  # Front View
#             ax.view_init(elev=0, azim=-90)
#             ax.set_title('Front View (YZ)')
#         elif event.key == 'd':  # Default 3D
#             ax.view_init(elev=30, azim=-60)
#             ax.set_title('Default 3D View')
#         plt.draw()  # Force redraw

#     # Connect the event to the figure
#     fig.canvas.mpl_connect('key_press_event', on_key)

#     print("Controls: Press 't' (Top), 's' (Side), 'f' (Front), 'd' (Default)")
#     plt.show()

# # --- ENTRY DATA ---
# if __name__ == "__main__":
#     input_data = {
#         'c_pivot': np.array([2.996, 0.142, 0.123]),
#         'c_fr': np.array([3.188, 0.125, 0.139]),
#         'c_rr': np.array([2.804, 0.158, 0.106]),
#         'pc_fr': np.array([3.094, 0.335, 0.136]),
#         'pc_fl': np.array([3.172, 0.246, 0.332]),
#         'pc_rr': np.array([2.712, 0.368, 0.103]),
#         'pc_rl': np.array([2.813, -0.114, 0.112]),
#         'epsilon': np.array([0.864, -0.058, 0.016, 0.5]),
#         'b_pos': np.array([2.982, 0.116, 0.258])
#     }

#     plot_scenario(input_data)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_best_fit_normal(points):
    """ Calculates best-fit plane normal using SVD """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    u, s, vh = np.linalg.svd(centered)
    normal = vh[2, :]
    if normal[2] < 0:
        normal = -normal
    return normal, centroid


def plot_real_scenario():
    # --- 1. Real Data Input ---
    pc_fr = np.array([0.103, 0.338, 0.048])  # Foot Front
    pc_rr = np.array([-0.283, 0.368, 0.043])  # Foot Rear
    c_fr = np.array([0.197, 0.128, 0.039])  # Shoulder Front
    c_rr = np.array([-0.189, 0.155, 0.044])  # Shoulder Rear

    # CoM (Flattened)
    r_pos = np.array([0.005, 0.148, 0.173])

    # Stack contact points for plane calculation
    contacts = np.array([pc_fr, pc_rr, c_fr, c_rr])

    # --- 2. Calculations ---
    # A. Best Fit Plane
    normal, centroid = get_best_fit_normal(contacts)

    # B. Pivot Axis (Shoulder to Shoulder)
    # This is the line the robot is likely rolling around
    pivot_center = (c_fr + c_rr) / 2

    # --- 3. Visualization Setup ---
    fig = plt.figure(figsize=(14, 6))

    # === SUBPLOT 1: 3D VIEW ===
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("3D View: Laying Down State")

    # Plot Points
    ax1.scatter(*pc_fr, c='k', s=50, label='Feet')
    ax1.scatter(*pc_rr, c='k', s=50)
    ax1.scatter(*c_fr, c='orange', s=80, marker='^', label='Shoulders (Pivots)')
    ax1.scatter(*c_rr, c='orange', s=80, marker='^')
    ax1.scatter(*r_pos, c='r', s=120, marker='s', label='CoM')

    # Draw Support Polygon (Cyan)
    # Order: c_rr -> c_fr -> pc_fr -> pc_rr (Counter-Clockwise)
    verts = [np.array([c_rr, c_fr, pc_fr, pc_rr])]
    poly = Poly3DCollection(verts, alpha=0.3, facecolors='cyan', edgecolors='b')
    ax1.add_collection3d(poly)

    # Draw Pivot Axis (Red Line)
    ax1.plot([c_rr[0], c_fr[0]], [c_rr[1], c_fr[1]], [c_rr[2], c_fr[2]], 'r-', linewidth=3, label='Pivot Axis')

    # Draw Gravity Vector from CoM
    # ax1.arrow(r_pos[0], r_pos[1], r_pos[2], 0)
    # ax1.text(r_pos[0], r_pos[1], r_pos[2] - 0.15, "mg", color='m')

    # Formatting
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    # Auto-scale
    all_pts = np.vstack([contacts, r_pos])
    mid = np.mean(all_pts, axis=0)
    dist = 0.3
    ax1.set_xlim(mid[0] - dist, mid[0] + dist)
    ax1.set_ylim(mid[1] - dist, mid[1] + dist)
    ax1.set_zlim(0, 0.4)
    ax1.view_init(elev=20, azim=130)  # View from "behind/right"

    # === SUBPLOT 2: TOP DOWN VIEW (XY) ===
    # This is the most important for checking stability margin
    ax2 = fig.add_subplot(122)
    ax2.set_title("Top-Down View (XY)\nStability Check")

    # Plot Points 2D
    ax2.scatter(pc_fr[0], pc_fr[1], c='k', label='Feet')
    ax2.scatter(pc_rr[0], pc_rr[1], c='k')
    ax2.scatter(c_fr[0], c_fr[1], c='orange', marker='^', s=80, label='Shoulders')
    ax2.scatter(c_rr[0], c_rr[1], c='orange', marker='^', s=80)
    ax2.scatter(r_pos[0], r_pos[1], c='r', marker='s', s=100, label='CoM')

    # Draw Pivot Line
    ax2.plot([c_rr[0], c_fr[0]], [c_rr[1], c_fr[1]], 'r--', linewidth=2, label='Pivot Axis')

    # Draw Stability Margin Line (Distance from CoM to Pivot Line)
    # We project CoM onto the line to see the gap
    # Simple visual line:
    ax2.plot([r_pos[0], r_pos[0]], [r_pos[1], c_fr[1]], 'm:', label='Y-margin')

    # Annotate coordinates for clarity
    ax2.text(r_pos[0] + 0.02, r_pos[1], f"CoM\nY={r_pos[1]:.3f}", color='r')
    ax2.text(c_fr[0] + 0.02, c_fr[1], f"Pivot_FR\nY={c_fr[1]:.3f}", color='orange')
    ax2.text(c_rr[0] - 0.15, c_rr[1], f"Pivot_RR\nY={c_rr[1]:.3f}", color='orange')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_real_scenario()
