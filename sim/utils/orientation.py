import numpy as np


def compute_alpha(epsilon):
    """
    Compute orientation alignment with gravity.

    epsilon: quaternion (x, y, z, w)
    returns: alpha in [0, 1]
    """

    qx, qy, qz, qw = epsilon

    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

    g_world = np.array([0.0, 0.0, -1.0])
    g_body = R.T @ g_world

    cos_alpha = -g_body[2]
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

    alpha = np.arccos(cos_alpha)
    alpha = 1.0 - alpha / np.pi

    return alpha
