import numpy as np


def contact_surfaces(c1, c2, c3, ref=np.array([1.0, 0.0, 0.0])):
    n = np.cross(c2 - c1, c3 - c1)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-10:
        return (np.array([0.0, 0.0, 1.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]))

    n = n / norm_n
    if n[2] < 0:
        n = -n

    t1 = ref - np.dot(ref, n) * n
    t1_norm = np.linalg.norm(t1)
    if t1_norm < 1e-10:          # only if the surface is vertical and ref ⊥ it
        t1 = np.cross(np.array([0.0, 0.0, 1.0]), n)
        t1 /= np.linalg.norm(t1)
    else:
        t1 = t1 / t1_norm

    t2 = np.cross(n, t1)         # already unit: n ⊥ t1, both unit
    return n, t1, t2
