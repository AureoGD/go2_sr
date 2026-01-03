import numpy as np
from scipy.spatial import ConvexHull


class SupportPolygonConstraint:
    """
    Compute linear constraints Ax <= b for the support polygon
    defined by contact / pivot points (XY projection).
    """

    @staticmethod
    def solve(contact_positions):
        """
        Args:
            contact_positions: Nx3 or Nx2 array of contact points

        Returns:
            A: (m x 2) matrix
            b: (m,) vector
            vertices: hull vertices in CCW order (for debugging / plotting)
        """

        # --------------------------------------------------
        # 1. Project to XY
        # --------------------------------------------------
        pts = contact_positions[:, :2]

        if len(pts) < 3:
            return None, None, None  # No polygon

        # --------------------------------------------------
        # 2. Convex hull
        # --------------------------------------------------
        try:
            hull = ConvexHull(pts)
            vertices = pts[hull.vertices]
        except Exception:
            return None, None, None

        # --------------------------------------------------
        # 3. Ensure CCW ordering
        # --------------------------------------------------
        if not SupportPolygonConstraint._is_ccw(vertices):
            vertices = vertices[::-1]

        # --------------------------------------------------
        # 4. Build half-space constraints
        # --------------------------------------------------
        A = []
        b = []

        n = len(vertices)
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            edge = v2 - v1

            # Inward normal (CCW polygon)
            normal = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue

            n_unit = normal / norm
            d = np.dot(n_unit, v1)

            A.append(n_unit)
            b.append(d)

        return np.array(A), np.array(b), vertices

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    @staticmethod
    def _is_ccw(vertices):
        area = 0.0
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        return area > 0
