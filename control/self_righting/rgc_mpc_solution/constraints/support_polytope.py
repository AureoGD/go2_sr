import numpy as np
from scipy.spatial import ConvexHull


def support_polytope(contacts):
    pts = np.asarray(contacts, dtype=float)[:, :2]
    verts = pts[ConvexHull(pts).vertices]
    centroid = verts.mean(axis=0)
    m = len(verts)
    A = np.empty((m, 2))
    b = np.empty(m)
    for i in range(m):  # <-- was range(m - 2)
        v0, v1 = verts[i], verts[(i + 1) % m]
        e = v1 - v0
        n = np.array([e[1], -e[0]]) / np.linalg.norm(e)
        if n @ (centroid - v0) > 0:
            n = -n
        A[i] = n
        b[i] = n @ v0

    if m< len(contacts):
        missing = len(contacts) - m
        A = np.vstack((A, np.zeros((missing,2))))
        b = np.concatenate((b, np.inf*np.ones((missing,))))

    return A, b
