import numpy as np


def plane_normal(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    u, s, vh = np.linalg.svd(centered_points)
    normal = vh[2, :]
    if normal[2] < 0:
        normal = -normal

    return normal, centroid
