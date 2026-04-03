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

        return J[:, 6:]

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

    def get_joint_limits(self):
        return self.joint_limits.copy()

    def get_torque_limits(self):
        return self.torque_limits.copy()
