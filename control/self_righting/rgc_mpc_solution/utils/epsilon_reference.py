import numpy as np
from control.self_righting.rgc_mpc_solution.utils.rotation_from_quat import rot_to_qua


def eps_reference(current_yaw=None, desired_yaw=None, plane_normal=None, pivot_direction=None, current_epsilon=None):
    if plane_normal is not None:
        desired_z = plane_normal / np.linalg.norm(plane_normal)
    else:
        desired_z = np.array([0., 0., 1.])

    if pivot_direction is not None:
        desired_x = pivot_direction / np.linalg.norm(pivot_direction)
    else:
        if desired_yaw is not None:
            yaw = desired_yaw
        elif current_yaw is not None:
            yaw = current_yaw
        else:
            yaw = 0
        desired_x = np.array([np.cos(yaw), np.sin(yaw), 0])

    desired_y = np.cross(desired_z, desired_x)
    if np.linalg.norm(desired_y) < 1e-6:
        desired_y = np.array([0., 1., 0.])
    else:
        desired_y = desired_y / np.linalg.norm(desired_y)

    desired_x = np.cross(desired_y, desired_z)
    desired_x = desired_x / np.linalg.norm(desired_x)

    R = np.column_stack([desired_x, desired_y, desired_z])
    q, R = rot_to_qua(R), R

    if current_epsilon is not None:
        if np.dot(current_epsilon, q) < 0:
            q = -q

    return q, R