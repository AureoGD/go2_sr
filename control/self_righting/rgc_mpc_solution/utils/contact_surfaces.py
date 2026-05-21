import numpy as np


def contact_surfaces(c1, c2, c3):
    v1 = c2 - c1
    v2 = c3 - c1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-10:
        n = np.array([0.0, 0.0, 1.0])
        t1 = np.array([1.0, 0.0, 0.0])
        t2 = np.array([0.0, 1.0, 0.0])
        return n, t1, t2

    if n[2] < 0:
        n = -n

    n = n / norm_n

    t1 = v1 / np.linalg.norm(v1)

    t1 = t1 - np.dot(t1, n) * n
    t1_norm = np.linalg.norm(t1)

    if t1_norm < 1e-10:

        if abs(n[0]) > 0.1 or abs(n[1]) > 0.1:
            t1 = np.array([-n[1], n[0], 0.0])
        else:
            t1 = np.array([1.0, 0.0, 0.0])
    else:
        t1 = t1 / t1_norm

    t2 = np.cross(n, t1)
    t2 = t2 / np.linalg.norm(t2)

    return n, t1, t2
