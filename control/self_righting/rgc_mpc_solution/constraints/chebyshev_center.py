import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


class ChebyshevCenterSolver:
    """
    MPC-grade Chebyshev center solver.
    - Fast
    - Robust
    - Convex
    """

    @staticmethod
    def solve(foot_positions, safety_margin=0.0):
        """
        Args:
            foot_positions: Nx3 array of contact / pivot positions
            safety_margin: shrink radius (meters)

        Returns:
            center_xy: np.array([x, y])
            radius: float
            A_hex, b_hex: hexagon approximation (optional MPC use)
        """

        # --------------------------------------------------
        # 1. Project to XY
        # --------------------------------------------------
        pts = foot_positions[:, :2]

        if len(pts) < 3:
            return ChebyshevCenterSolver._degenerate_case(pts)

        # --------------------------------------------------
        # 2. Convex hull
        # --------------------------------------------------
        try:
            hull = ConvexHull(pts)
            vertices = pts[hull.vertices]
        except Exception:
            return ChebyshevCenterSolver._degenerate_case(pts)

        # Ensure CCW
        if not ChebyshevCenterSolver._is_ccw(vertices):
            vertices = vertices[::-1]

        # --------------------------------------------------
        # 3. Build half-space constraints: A x + r ≤ b
        # --------------------------------------------------
        A = []
        b = []

        n = len(vertices)
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            edge = v2 - v1

            normal = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue

            n_unit = normal / norm
            d = np.dot(n_unit, v1)

            A.append([n_unit[0], n_unit[1], 1.0])
            b.append(d)

        A = np.array(A)
        b = np.array(b)

        # --------------------------------------------------
        # 4. Linear program: maximize r
        # --------------------------------------------------
        # linprog minimizes, so minimize -r
        c = np.array([0.0, 0.0, -1.0])

        res = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None), (None, None), (0, None)], method="highs")

        if not res.success:
            return ChebyshevCenterSolver._fallback(vertices)

        x, y, r = res.x
        r = max(0.0, r - safety_margin)

        # --------------------------------------------------
        # 5. Hexagon approximation (QP-friendly)
        # --------------------------------------------------
        A_hex, b_hex = ChebyshevCenterSolver._hexagon_constraints(center=np.array([x, y]), radius=r)

        return np.array([x, y]), r, A_hex, b_hex

    # ==================================================
    # Helpers
    # ==================================================

    @staticmethod
    def _hexagon_constraints(center, radius):
        """
        Returns A, b such that A (p - center) ≤ b
        """
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        normals = np.c_[np.cos(angles), np.sin(angles)]

        A = normals
        b = radius + A @ center

        return A, b

    @staticmethod
    def _is_ccw(vertices):
        area = 0.0
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        return area > 0

    @staticmethod
    def _degenerate_case(pts):
        center = np.mean(pts, axis=0) if len(pts) > 0 else np.zeros(2)
        return center, 0.0, None, None

    @staticmethod
    def _fallback(vertices):
        center = np.mean(vertices, axis=0)
        return center, 0.0, None, None
