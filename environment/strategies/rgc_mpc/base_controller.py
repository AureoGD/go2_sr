import os
import warnings
from contextlib import contextmanager

import numpy as np
import pinocchio as pin
import osqp
from scipy import sparse

from environment.strategies.rgc_mpc.chebyshev_center import ChebyshevCenterSolver


@contextmanager
def silence_all_output():
    """
    Silences BOTH Python warnings and C-level library output (OSQP, MuJoCo).
    """
    # 1. Silence Python Warnings (RuntimeWarning: overflow, etc.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 2. Silence C-Level Stdout/Stderr (OSQP validation errors)
        try:
            # Open null file
            null_fd = os.open(os.devnull, os.O_RDWR)

            # Save actual file descriptors
            save_stdout = os.dup(1)
            save_stderr = os.dup(2)

            # Redirect to null
            os.dup2(null_fd, 1)
            os.dup2(null_fd, 2)

            yield

        finally:
            # Restore original streams
            os.dup2(save_stdout, 1)
            os.dup2(save_stderr, 2)

            # Clean up
            os.close(null_fd)
            os.close(save_stdout)
            os.close(save_stderr)


class LogCompletionDetector:
    """
    Simple log-space completion detector.
    """

    def __init__(self, window_size=20, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.log_history = []

    def is_task_complete(self, current_obj_val):
        abs_val = abs(current_obj_val)
        epsilon = 1e-10
        current_log = np.log10(abs_val + epsilon)

        self.log_history.append(current_log)

        if len(self.log_history) > self.window_size:
            self.log_history.pop(0)

        if len(self.log_history) < self.window_size:
            return False

        recent_logs = np.array(self.log_history)
        log_range = np.max(recent_logs) - np.min(recent_logs)

        return log_range < self.threshold

    def reset(self):
        self.log_history = []


class BaseRGC:

    TASK_NAME = "undefined"
    TASK_LEVEL = -1

    def __init__(self, **kwargs):
        # Always set semantic metadata
        self.task_name = self.TASK_NAME
        self.task_level = self.TASK_LEVEL
        self.task_finish_detect = None

        # Detect runtime vs metadata-only mode
        self.runtime = "robot_states" in kwargs

        if not self.runtime:
            return

        self.model = kwargs.get('model')
        self.data = kwargs.get('data')
        self.geo_model = kwargs.get('geo_model')
        self.geo_data = kwargs.get('geo_data')
        self.links = kwargs.get('links')
        self.legs = kwargs.get('legs')
        self.links_ids = kwargs.get('links_ids')
        self.foot_ids = kwargs.get('foot_ids')
        self.kp = kwargs.get('kp')
        self.kd = kwargs.get('kd')
        self.robot_states = kwargs.get('robot_states', None)
        self.total_mass = self.get_total_mass()
        # Give a name for each task and a "level"

        self.i = 0
        self.N = None
        self.M = None
        self.ts = None
        self.A = None
        self.B = None
        self.Aa = None
        self.Ba = None
        self.Ca = None
        self.C_cons = None
        self.ref = None
        self.nu = None
        self.nx = None
        self.ny = None
        self.nc = None
        self.Q = None
        self.R = None
        self.x = None
        self.l = None
        self.u = None
        self.prob = osqp.OSQP()
        self.op_init = False
        self.first_int = True
        self.min_obj_val = 100

        self.convergence_threshold = 0.15
        self.ws = 20

        self.center_optimizer = ChebyshevCenterSolver()

    def _update_detector(self):
        self.task_finish_detect = LogCompletionDetector(window_size=self.ws, threshold=self.convergence_threshold)

    def update_dqr(self):
        if not self.runtime:
            return np.zeros(12)
        try:
            # Silence ALL outputs (NumPy warnings, OSQP C-logs, etc.)
            with silence_all_output():

                # 1. Update Model (Populates self.x)
                self.update_model()

                # 2. SAFETY CHECK: Check validity NOW, after update
                # Handle None (initialization error) or NaN/Inf (divergence)
                if self.x is None:
                    return self._handle_mpc_failure(critical=True)

                if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)):
                    return self._handle_mpc_failure(critical=True)

                # 3. Prediction & Matrices
                Phi, G, Phi_cons, G_cons = self.update_pred_mdl()

                H_dense = G.T @ self.Q @ G + self.R
                F_dense = 2 * (((Phi @ self.x) - self.ref).T) @ self.Q @ G

                # Convert to sparse
                H = 2 * sparse.csc_matrix(H_dense)
                G_cons_sparse = sparse.csc_matrix(G_cons)
                F = F_dense

                self.prob = osqp.OSQP()

                # --- Setup OSQP ---
                try:
                    self.prob.setup(H,
                                    F.T,
                                    A=G_cons_sparse,
                                    l=self.l - Phi_cons @ self.x,
                                    u=self.u - Phi_cons @ self.x,
                                    verbose=False,
                                    warm_start=True,
                                    max_iter=2000,
                                    check_termination=10)
                except ValueError:
                    # Configuration Failure (e.g. Lower Bound > Upper Bound)
                    return self._handle_mpc_failure(critical=True)

                # --- Solve ---
                res = self.prob.solve()

                if res.info.status != "solved":
                    # Solver Failure (not critical, just sub-optimal)
                    return self._handle_mpc_failure(critical=False)
                else:
                    return self._handle_mpc_success(res)

        except Exception:
            # Catch-all for unexpected errors
            return self._handle_mpc_failure(critical=True)

    def _handle_mpc_failure(self, critical=True):
        """
        Handles failure states.
        - Always sets mpc_fail = True
        - critical=True means configuration/setup failed (stop episode)
        - critical=False means solver couldn't find optimal solution (maybe continue?)
        """
        self.robot_states.mpc_fail = True
        self.robot_states.critical_mpc_fail = critical
        self.dqr = np.zeros((self.nu,), dtype=np.float32)
        return self.dqr

    def _handle_mpc_success(self, res):
        """Helper to handle MPC success."""
        is_complete = self.task_finish_detect.is_task_complete(abs(res.info.obj_val))
        if is_complete:
            pass
        self.robot_states.subtask_succes = is_complete

        self.robot_states.mpc_obj_val = res.info.obj_val
        self.robot_states.mpc_fail = False
        self.robot_states.critical_mpc_fail = False
        self.dqr = res.x[0:self.nu]
        return self.dqr

    def update_model(self):
        raise NotImplementedError("Subclasses must implement update_model() method")

    def define_constraints_matrices(self):
        raise NotImplementedError("Subclasses must implement define_constraints_matrices() method")

    def update_pred_mdl(self):
        G = np.zeros((self.ny * self.N, self.nu * self.M))
        Phi = np.zeros((self.ny * self.N, self.nx + self.nu))
        aux = np.zeros((self.ny, self.nu))

        # Calculate initial blocks
        aux[:, :] = self.Ca @ self.Ba
        Phi[0:self.ny, :] = self.Ca @ self.Aa

        G_cons = np.zeros((self.nc * self.N, self.nu * self.M))
        aux_cons, Phi_cons = self.define_constraints_matrices()

        for i in range(self.N):
            j = 0
            if i != 0:
                # Prediction propagation
                Phi[i * self.ny:(i + 1) * self.ny, :] = Phi[(i - 1) * self.ny:i * self.ny, :] @ self.Aa
                aux[:, :] = Phi[(i - 1) * self.ny:i * self.ny, :] @ self.Ba

                Phi_cons[i * self.nc:(i + 1) * self.nc, :] = Phi_cons[(i - 1) * self.nc:i * self.nc, :] @ self.Aa
                aux_cons[:, :] = Phi_cons[(i - 1) * self.nc:i * self.nc, :] @ self.Ba

            while (j < self.M) and (i + j < self.N):
                G[(i + j) * self.ny:(i + j + 1) * self.ny, j * (self.nu):(j + 1) * (self.nu)] = aux[:, :]
                G_cons[(i + j) * self.nc:(i + j + 1) * self.nc, j * (self.nu):(j + 1) * (self.nu)] = aux_cons[:, :]
                j += 1

        return Phi, G, Phi_cons, G_cons

    def get_total_mass(self):
        total_mass = 0.0
        for inertia in self.model.inertias:
            total_mass += inertia.mass
        return total_mass

    def skew_symmetric_matrix(self, vector):
        v1, v2, v3 = vector
        matrix = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])
        return matrix

    def ordering_joints(self):
        """
        This method ordering the robot joint postions and velocities 
        according to the pinocchio's framework
        """
        q = np.vstack((self.robot_states.b_pos, self.robot_states.epsilon, self.robot_states.q[3:6],
                       self.robot_states.q[0:3], self.robot_states.q[9:12], self.robot_states.q[6:9]))

        dq = np.vstack((self.robot_states.b_vel, self.robot_states.omega, self.robot_states.dq[3:6],
                        self.robot_states.dq[0:3], self.robot_states.dq[9:12], self.robot_states.dq[6:9]))

        return q, dq

    def eps_reference(self, current_yaw=None, desired_yaw=None):
        """
        Compute quaternion that makes torso vertical while preserving yaw
        """
        gravity = np.array([0, 0, -1])

        desired_z = -gravity  # = [0, 0, 1]

        if desired_yaw is not None:
            yaw = desired_yaw
        elif current_yaw is not None:
            yaw = current_yaw  # Maintain current yaw
        else:
            yaw = 0  # Default

        desired_x = np.array([np.cos(yaw), np.sin(yaw), 0])

        desired_y = np.cross(desired_z, desired_x)
        desired_y = desired_y / np.linalg.norm(desired_y)
        desired_x = np.cross(desired_y, desired_z)

        R = np.column_stack([desired_x, desired_y, desired_z])

        eps_ref = self.rotation_matrix_to_quaternion(R)

        return eps_ref

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a 3x3 rotation matrix to a quaternion [x, y, z, w]
        """
        R = np.asarray(R)
        q = np.zeros(4)

        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2  # S = 4 * qw
            q[3] = 0.25 * S  # w
            q[0] = (R[2, 1] - R[1, 2]) / S  # x
            q[1] = (R[0, 2] - R[2, 0]) / S  # y
            q[2] = (R[1, 0] - R[0, 1]) / S  # z

        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S = 4 * qx
            q[3] = (R[2, 1] - R[1, 2]) / S  # w
            q[0] = 0.25 * S  # x
            q[1] = (R[0, 1] + R[1, 0]) / S  # y
            q[2] = (R[0, 2] + R[2, 0]) / S  # z

        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S = 4 * qy
            q[3] = (R[0, 2] - R[2, 0]) / S  # w
            q[0] = (R[0, 1] + R[1, 0]) / S  # x
            q[1] = 0.25 * S  # y
            q[2] = (R[1, 2] + R[2, 1]) / S  # z

        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S = 4 * qz
            q[3] = (R[1, 0] - R[0, 1]) / S  # w
            q[0] = (R[0, 2] + R[2, 0]) / S  # x
            q[1] = (R[1, 2] + R[2, 1]) / S  # y
            q[2] = 0.25 * S  # z

        q = q / np.linalg.norm(q)
        return q

    def com_quatities(self):
        q, dq = self.ordering_joints()
        pin.ccrba(self.model, self.data, q, dq)

        r = self.data.com[0]
        dr = self.data.vcom[0]

        # Save states
        self.robot_states.r_vel = dr.reshape(3, 1)
        self.robot_states.r_pos = r.reshape(3, 1)

    def reset_controller(self):
        if self.task_finish_detect is not None:
            self.task_finish_detect.reset()
        self.first_int = True
        self.prob = None
