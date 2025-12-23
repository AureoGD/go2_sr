import numpy as np
from scipy.spatial import ConvexHull


class ChebyshevCenterSolver:
    """
    Minimal, correct Chebyshev center solver.
    Input: foot_positions (Nx3 array)
    Output: (center, radius) where center = (x, y), radius >= 0
    """

    @staticmethod
    def solve(foot_positions):
        """
        Compute Chebyshev center and maximum inscribed radius.
        
        Args:
            foot_positions: Nx3 array of [x, y, z] foot positions
            
        Returns:
            tuple: ((center_x, center_y), radius_in_meters)
        """
        # Project to XY plane
        points = foot_positions[:, :2]

        # Handle degenerate cases
        if len(points) < 3:
            return ChebyshevCenterSolver._degenerate_case(points)

        # Compute convex hull
        try:
            hull = ConvexHull(points)
        except:
            return ChebyshevCenterSolver._degenerate_case(points)

        # Get vertices in CCW order
        vertices = points[hull.vertices]
        if not ChebyshevCenterSolver._is_ccw(vertices):
            vertices = vertices[::-1]

        # Compute edge constraints with unit normals
        edges = []
        n = len(vertices)

        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            edge = v2 - v1

            # Inward normal for CCW polygon (rotate edge 90° clockwise)
            n_vec = np.array([edge[1], -edge[0]])
            norm = np.linalg.norm(n_vec)

            if norm < 1e-10:
                continue

            n_unit = n_vec / norm  # Unit normal
            d = np.dot(n_unit, v1)  # Plane constant

            edges.append({'n': n_unit, 'd': d, 'v1': v1, 'v2': v2})

        # Solve Chebyshev: maximize r subject to n_i·x + r ≤ d_i
        best_center, best_radius = None, -np.inf
        m = len(edges)

        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    try:
                        # Solve linear system: n·x + r = d
                        A = np.array([[edges[i]['n'][0], edges[i]['n'][1], 1.0],
                                      [edges[j]['n'][0], edges[j]['n'][1], 1.0],
                                      [edges[k]['n'][0], edges[k]['n'][1], 1.0]])
                        b = np.array([edges[i]['d'], edges[j]['d'], edges[k]['d']])

                        x, y, r = np.linalg.solve(A, b)

                        if r < 0:
                            continue

                        # Verify feasibility
                        feasible = True
                        for edge in edges:
                            if edge['n'][0] * x + edge['n'][1] * y + r > edge['d'] + 1e-8:
                                feasible = False
                                break

                        if feasible and r > best_radius:
                            best_radius = r
                            best_center = np.array([x, y])

                    except np.linalg.LinAlgError:
                        continue

        # Fallback to centroid if no solution
        if best_center is None:
            return ChebyshevCenterSolver._centroid_fallback(edges)

        return best_center, max(0, best_radius)

    @staticmethod
    def _is_ccw(vertices):
        """Check CCW orientation using shoelace formula"""
        area = 0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        return area > 0

    @staticmethod
    def _centroid_fallback(edges):
        """Fallback using polygon centroid"""
        if not edges:
            return np.array([0.0, 0.0]), 0.0

        vertices = np.array([e['v1'] for e in edges])
        centroid = np.mean(vertices, axis=0)

        # Find minimum distance to edges
        distances = []
        for edge in edges:
            dist = np.dot(edge['n'], centroid) - edge['d']
            distances.append(dist)

        radius = max(0, -min(distances))  # Negative distance means outside
        return centroid, radius * 0.8  # 20% safety margin

    @staticmethod
    def _degenerate_case(points):
        """Handle <3 points or collinear points"""
        if len(points) == 0:
            return np.array([0.0, 0.0]), 0.0
        elif len(points) == 1:
            return points[0], 0.0
        elif len(points) == 2:
            center = np.mean(points, axis=0)
            radius = np.linalg.norm(points[0] - points[1]) / 4
            return center, radius

        # For 3+ points, use bounding box
        center = np.mean(points, axis=0)
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        radius = min(x_max - x_min, y_max - y_min) / 4

        return center, max(0, radius)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Your data
    foot_positions = np.array([
        [2.159, 1.897, 0.312],  # FR
        [2.087, 2.174, 0.301],  # FL
        [1.794, 1.795, 0.251],  # RR
        [1.719, 2.080, 0.242]  # RL
    ])

    # Solve
    center, radius = ChebyshevCenterSolver.solve(foot_positions)

    print(f"Chebyshev Center: ({center[0]:.4f}, {center[1]:.4f})")
    print(f"Chebyshev Radius: {radius:.4f} m")
    print(f"Safe margin for CoM: ±{radius*100:.1f} cm")

    # With safety margin
    safety_margin = 0.05  # 5 cm
    safe_radius = max(0, radius - safety_margin)
    print(f"\nWith {safety_margin*100:.0f} cm safety margin:")
    print(f"  Safe radius: {safe_radius:.4f} m")
    print(f"  Safe CoM movement: ±{safe_radius*100:.1f} cm")

    # Quick performance test
    import time
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        ChebyshevCenterSolver.solve(foot_positions)
        times.append(time.perf_counter() - start)

    avg_time_ms = np.mean(times) * 1000
    print(f"\nPerformance: {avg_time_ms:.3f} ms per solve")
    print(f"Can run at {int(1000/avg_time_ms):d} Hz")
