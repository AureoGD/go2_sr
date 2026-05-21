import numpy as np


def rot_to_qua(R):
    """
        Convert a 3x3 rotation matrix to a quaternion [x, y, z, w]
        """
    R = np.asarray(R)
    q = np.zeros(4)

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * qw
        q[3] = 0.25 * S  # w
        q[0] = (R[2, 1] - R[1, 2]) / S  # x
        q[1] = (R[0, 2] - R[2, 0]) / S  # y
        q[2] = (R[1, 0] - R[0, 1]) / S  # z

    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
        q[3] = (R[2, 1] - R[1, 2]) / S  # w
        q[0] = 0.25 * S  # x
        q[1] = (R[0, 1] + R[1, 0]) / S  # y
        q[2] = (R[0, 2] + R[2, 0]) / S  # z

    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
        q[3] = (R[0, 2] - R[2, 0]) / S  # w
        q[0] = (R[0, 1] + R[1, 0]) / S  # x
        q[1] = 0.25 * S  # y
        q[2] = (R[1, 2] + R[2, 1]) / S  # z

    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
        q[3] = (R[1, 0] - R[0, 1]) / S  # w
        q[0] = (R[0, 2] + R[2, 0]) / S  # x
        q[1] = (R[1, 2] + R[2, 1]) / S  # y
        q[2] = 0.25 * S  # z

    q = q / np.linalg.norm(q)

    return q
