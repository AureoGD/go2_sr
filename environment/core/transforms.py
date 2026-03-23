import numpy as np
from scipy.spatial.transform import Rotation as R


def euler_to_quat(rpy, order='wxyz'):
    """
    rpy: array-like (roll, pitch, yaw)
    order: 'wxyz' (MuJoCo) or 'xyzw' (Pinocchio)
    """
    quat_xyzw = R.from_euler('xyz', rpy).as_quat()

    if order == 'xyzw':
        return quat_xyzw

    elif order == 'wxyz':
        return np.array([quat_xyzw[3], *quat_xyzw[:3]])

    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'")


def quat_to_euler(quat, order='wxyz'):
    """
    quat: quaternion
    order: input format
    """
    if order == 'wxyz':
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    elif order == 'xyzw':
        quat_xyzw = quat
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'")

    return R.from_quat(quat_xyzw).as_euler('xyz')
