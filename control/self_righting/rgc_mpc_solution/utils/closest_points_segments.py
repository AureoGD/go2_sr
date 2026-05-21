import numpy as np


def closest_points_segments(p1a, p1b, p2a, p2b):
    """
        Finds the closest points on two segments P1 and P2.
        P1 = p1a + s*(p1b-p1a)
        P2 = p2a + t*(p2b-p2a)
        Returns: (point_on_1, point_on_2, distance, normal)
        """
    # Vectors direction of the segments
    d1 = p1b - p1a
    d2 = p2b - p2a
    r = p1a - p2a

    # Squared lengths
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    # Check for degenerate segments (points)
    if a <= 1e-6 and e <= 1e-6:
        # Both segments are points
        return p1a, p2a, np.linalg.norm(p1a - p2a), (p1a - p2a)

    # Standard Case
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b

    # If segments not parallel, compute closest point on infinite lines
    if denom != 0.0:
        s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
    else:
        s = 0.0  # Parallel lines, pick start

    # Compute t based on s
    t = (b * s + f) / e

    # If t is out of bounds, clamp it and re-evaluate s
    if t < 0.0:
        t = 0.0
        s = np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - c) / a, 0.0, 1.0)

    # Calculate closest points
    c1 = p1a + s * d1
    c2 = p2a + t * d2

    # Distance and Normal
    diff = c1 - c2
    dist = np.linalg.norm(diff)

    if dist > 1e-6:
        n = diff / dist
    else:
        n = np.array([0, 0, 1])  # Default normal if overlapping

    return c1, c2, dist, n
