import numpy as np

from control.self_righting.rgc_mpc_solution.utils.closest_points_segments import (
    closest_points_segments,)


def self_collision_constraints(
    pin_engine,
    pairs,
    radius,
    d_safe,
):
    """
    Builds self-collision avoidance constraints between leg pairs.

    Parameters
    ----------
    pin_engine : PinocchioEngine
        Shared robot kinematics/dynamics interface.

    pairs : list[tuple[str, str]]
        Collision pairs.
        Example:
            [
                ("FR", "FL"),
                ("FR", "RR"),
                ...
            ]

    radius : float
        Capsule/segment radius.

    d_safe : float
        Additional safety distance.

    Returns
    -------
    J : np.ndarray shape (n_pairs, 12)
        Constraint Jacobian matrix.

    dist : np.ndarray shape (n_pairs, 1)
        Minimum distance constraint vector.
    """

    # -------------------------------------------------
    # Constraint Jacobian
    # -------------------------------------------------
    J = np.zeros((len(pairs), 12))

    # Minimum allowed distance
    min_dist = d_safe + 2.0 * radius

    dist_list = []

    # -------------------------------------------------
    # Evaluate all collision pairs
    # -------------------------------------------------
    for row, (leg1, leg2) in enumerate(pairs):

        # Segment endpoints
        foot1 = pin_engine.frame_pos(leg1, "foot")
        calf1 = pin_engine.frame_pos(leg1, "calf")

        foot2 = pin_engine.frame_pos(leg2, "foot")
        calf2 = pin_engine.frame_pos(leg2, "calf")

        # Closest points and separating normal
        c1, c2, pair_dist, n = closest_points_segments(calf1, foot1, calf2, foot2)

        # Point Jacobians
        J1 = pin_engine.point_jacobian(leg1, "foot", c1)

        J2 = pin_engine.point_jacobian(leg2, "foot", c2)

        # Joint block slices
        s1 = pin_engine.leg_slices[leg1]
        s2 = pin_engine.leg_slices[leg2]

        # Constraint Jacobian row
        J[row, s1] = n @ J1
        J[row, s2] = -n @ J2

        # Distance constraint
        dist_list.append(pair_dist)

    # -------------------------------------------------
    # Distance vector
    # -------------------------------------------------
    dist = min_dist - np.array(dist_list).reshape(-1, 1)

    return J, dist
