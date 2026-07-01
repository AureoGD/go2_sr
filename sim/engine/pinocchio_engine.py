import numpy as np
import pinocchio as pin


class PinocchioEngine:

    # ======================================================
    # INIT
    # ======================================================
    def __init__(self, model):

        self.model = model
        self.data = model.createData()

        # -------------------------------
        # Ordem canônica (Unitree)
        # -------------------------------
        self.leg_order = ["FR", "FL", "RR", "RL"]

        # Ordem do Pinocchio (URDF)
        self.pin_leg_order = ["FL", "FR", "RL", "RR"]

        # Mapping Unitree → Pinocchio
        self.leg_map = {leg: self.pin_leg_order.index(leg) for leg in self.leg_order}

        # -------------------------------
        # Frames (semântica → nome)
        # -------------------------------
        self.frame_names = {}

        for leg in self.leg_order:
            self.frame_names[leg] = {
                "hip": f"{leg}_hip_joint",
                "thigh": f"{leg}_thigh_joint",
                "calf": f"{leg}_calf_joint",
                "foot": f"{leg}_foot",
            }

        # -------------------------------
        # Frames (nome → id)
        # -------------------------------
        self.frames = {}

        for leg in self.leg_order:
            self.frames[leg] = {}

            for part, name in self.frame_names[leg].items():

                if not self.model.existFrame(name):
                    raise ValueError(f"Frame {name} não existe no modelo")

                self.frames[leg][part] = self.model.getFrameId(name)

        self.leg_slices = {
            "FR": slice(0, 3),
            "FL": slice(3, 6),
            "RR": slice(6, 9),
            "RL": slice(9, 12),
        }

        self.order = [
            3,
            4,
            5,  # FR
            0,
            1,
            2,  # FL
            9,
            10,
            11,  # RR
            6,
            7,
            8  # RL
        ]

        # -------------------------------
        # JOINT & TORQUE LIMITS
        # -------------------------------
        self._extract_joint_limits()

        # cache para acesso direto por string
        self._frame_cache = {}

        # estado interno
        self.q = None
        self.dq = None
        self.updated = False

        self.robot_mass = self._total_mass()

    # ======================================================
    # BUILD q, dq (Unitree → Pinocchio)
    # ======================================================
    def _build_q_dq(self, rs):

        q_legs = []
        dq_legs = []

        for leg in self.pin_leg_order:
            idx = self.leg_order.index(leg)

            q_legs.append(rs.q[idx * 3:(idx + 1) * 3])
            dq_legs.append(rs.dq[idx * 3:(idx + 1) * 3])

        q_legs = np.concatenate(q_legs)
        dq_legs = np.concatenate(dq_legs)

        q = np.concatenate([
            rs.b_pos,
            rs.epsilon,  # quaternion wxyz
            q_legs
        ])

        dq = np.concatenate([rs.b_vel, rs.omega, dq_legs])

        return q, dq

    # ======================================================
    # UPDATE
    # ======================================================
    def update(self, robot_state):

        self.q, self.dq = self._build_q_dq(robot_state)

        pin.forwardKinematics(self.model, self.data, self.q, self.dq)
        pin.updateFramePlacements(self.model, self.data)

        pin.crba(self.model, self.data, self.q)
        pin.computeCoriolisMatrix(self.model, self.data, self.q, self.dq)
        pin.ccrba(self.model, self.data, self.q, self.dq)
        pin.centerOfMass(self.model, self.data, self.q, self.dq)

        self.updated = True

    def _check(self):
        if not self.updated:
            raise RuntimeError("PinocchioEngine.update() não foi chamado")

    def centroidal_inertia(self):
        self._check()
        return self.data.Ig.inertia.copy()

    # ======================================================
    # REORDER (Pinocchio → Unitree)
    # ======================================================
    def reorder_legs(self, X, block_size=3):

        blocks = [X[i * block_size:(i + 1) * block_size] for i in range(4)]

        reordered = [blocks[self.leg_map[leg]] for leg in self.leg_order]

        return np.concatenate(reordered, axis=0)

    # ======================================================
    # CoM GLOBAL
    # ======================================================
    def com(self):
        self._check()
        return self.data.com[0].copy()

    def vcom(self):
        self._check()
        return self.data.vcom[0].copy()

    # ======================================================
    # JACOBIANO DO CoM
    # ======================================================
    def com_jacobian(self):
        self._check()

        J = pin.jacobianCenterOfMass(self.model, self.data, self.q, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        J_joints = J[:, 6:]

        return self.reorder_legs(J_joints.T).T

    def com_jacobian_legs(self):
        J = self.com_jacobian()

        return {leg: J[:, i * 3:(i + 1) * 3] for i, leg in enumerate(self.leg_order)}

    # ======================================================
    # FRAME POSITION (API híbrida)
    # ======================================================
    def frame_pos(self, leg, part=None):
        self._check()

        # modo semântico (recomendado)
        if part is not None:
            fid = self.frames[leg][part]
            return self.data.oMf[fid].translation.copy()

        # modo string direta ("FR_foot")
        name = leg

        if name not in self._frame_cache:
            self._frame_cache[name] = self.model.getFrameId(name)

        fid = self._frame_cache[name]

        return self.data.oMf[fid].translation.copy()

    # ======================================================
    # FRAME JACOBIAN (semântico)
    # ======================================================
    def frame_jacobian(self, leg, part):
        self._check()

        fid = self.frames[leg][part]

        J = pin.computeFrameJacobian(self.model, self.data, self.q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        J_joints = J[:, 6:]

        return self.reorder_legs(J_joints.T).T

    # ======================================================
    # HELPERS
    # ======================================================
    def leg_positions(self, leg):
        return {part: self.frame_pos(leg, part) for part in ["hip", "thigh", "calf", "foot"]}

    def feet_positions_array(self):
        return np.array([self.frame_pos(leg, "foot") for leg in self.leg_order])

    def feet_jacobians(self):
        return {leg: self.frame_jacobian(leg, "foot") for leg in self.leg_order}

    # ======================================================
    # DYNAMICS
    # ======================================================
    def gravity(self):
        self._check()

        tau = pin.computeGeneralizedGravity(self.model, self.data, self.q)[6:]

        return self.reorder_legs(tau)

    def _extract_joint_limits(self):

        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        effort = self.model.effortLimit

        joint_lower = lower[7:]
        joint_upper = upper[7:]
        joint_effort = effort[6:]

        joint_lower = self.reorder_legs(joint_lower)
        joint_upper = self.reorder_legs(joint_upper)
        joint_effort = self.reorder_legs(joint_effort)

        self.joint_limits = np.stack([joint_lower, joint_upper], axis=1)
        self.torque_limits = joint_effort

    def actuated_mass_matrix(self):
        """
        Returns the actuated joint-space mass matrix
        reordered to the controller/Unitree convention.

        Output shape:
            (12, 12)
        """

        # Full floating-base mass matrix
        M_full = self.data.M.copy()

        # Remove floating base (first 6 DoFs)
        M_act = M_full[6:, 6:]

        # Reorder joints from URDF/Pinocchio ordering
        # to controller ordering

        M_reordered = M_act[np.ix_(self.order, self.order)]

        return M_reordered

    def linear_leg_jacobian(self, leg, part):

        J_frame = self.frame_jacobian(leg, part)

        J_linear = J_frame[:3, :]

        joint_slice = self.leg_slices[leg]

        return J_linear[:, joint_slice]

    def angular_leg_jacobian(self, leg, part):

        J_frame = self.frame_jacobian(leg, part)

        J_linear = J_frame[3:, :]

        joint_slice = self.leg_slices[leg]

        return J_linear[:, joint_slice]

    def point_jacobian(self, leg_name, frame_name, point_world):
        """
        Returns the 3x3 linear Jacobian block evaluated
        at an arbitrary point expressed in world coordinates.

        Parameters
        ----------
        leg_name : str
            Leg identifier:
                "FR", "FL", "RR", "RL"

        frame_name : str
            Frame suffix name:
                "foot", "calf_joint", etc.

        point_world : np.ndarray shape (3,)
            Point expressed in world coordinates.

        Returns
        -------
        np.ndarray shape (3,3)
            Linear Jacobian block associated with the leg.
        """

        # -------------------------------------------------
        # Build full frame name
        # -------------------------------------------------
        full_frame_name = f"{leg_name}_{frame_name}"

        frame_id = self.model.getFrameId(full_frame_name)

        # -------------------------------------------------
        # Get frame placement
        # -------------------------------------------------
        oMf = self.data.oMf[frame_id]

        # Vector from frame origin to target point
        r_vec = point_world - oMf.translation

        # -------------------------------------------------
        # Compute frame Jacobian
        # -------------------------------------------------
        J_frame = pin.computeFrameJacobian(self.model, self.data, self.q, frame_id, pin.LOCAL_WORLD_ALIGNED)

        # Split linear/angular components
        J_linear = J_frame[:3, :]
        J_angular = J_frame[3:, :]

        # -------------------------------------------------
        # Point Jacobian transformation
        # -------------------------------------------------
        J_point = J_linear - pin.skew(r_vec) @ J_angular

        # Remove floating base
        J_point = J_point[:, 6:]

        # -------------------------------------------------
        # Reorder to controller convention
        # -------------------------------------------------

        J_point = J_point[:, self.order]

        # -------------------------------------------------
        # Return only desired leg block
        # -------------------------------------------------
        joint_slice = self.leg_slices[leg_name]

        return J_point[:, joint_slice]

    def get_base_rot_mtx(self):
        base_id = self.model.getFrameId("base_link")
        pin.updateFramePlacements(self.model, self.data)

        R_base = np.asarray(self.data.oMf[base_id].rotation)  # (3,3)
        p_base = np.asarray(self.data.oMf[base_id].translation)  # (3,)

        return R_base

    def get_joint_limits(self):
        return self.joint_limits.copy()

    def get_torque_limits(self):
        return self.torque_limits.copy()

    def _total_mass(self):
        return pin.computeTotalMass(self.model)
