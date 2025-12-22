import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import math
import pinocchio as pin
from environment.robot_states import RobotStates
from scipy.spatial.transform import Rotation
from sr_strategies.tsm.unitree_sr import UnitreeSR
from sr_strategies.rgc.controller_screduller import ControlScheduler
# from sr_strategies.rgc.controller_screduller_backup import ControlScheduler
from sr_strategies.rgc.base_controller import BaseRGC
import itertools

np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=3)


class Go2ModelSimMuJoCo():

    def __init__(self, task_control=None, render=False):
        # system dynamics and control sample time
        self._is_render = render
        self.pin_model = None
        self.pin_data = None
        self.geo_data = None
        self.geo_model = None
        self.task_control = task_control

        self.links_ids = []
        self.foot_ids = []
        self.joint_idx_list = []

        self.legs = ['FR', 'FL', 'RR', 'RL']  # Your required order
        self.links = ['hip', 'thigh', 'calf']

        self.dyn_dt = 0.001
        self.con_dt = 0.01

        # Load MuJoCo model
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unitree_go2/scene.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL

        # Initialize pinocchio model
        urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unitree_go2/go2.urdf")
        if os.path.exists(urdf_path):
            root_joint = pin.JointModelFreeFlyer()
            self.pin_model = pin.buildModelFromUrdf(urdf_path, root_joint)
            self.pin_data = self.pin_model.createData()
            self.geo_model = pin.buildGeomFromUrdf(self.pin_model, urdf_path, pin.GeometryType.COLLISION)
        else:
            print("Warning: URDF file not found, Pinocchio model not loaded")

        # self.setup_collision_pairs()
        self.geom_data = pin.GeometryData(self.geo_model)

        # Set up joint and actuator mapping with correct order
        self._setup_joint_mapping()

        # Physics parameters
        self.model.opt.timestep = self.dyn_dt

        self.viewer = None
        if self._is_render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        # Control gains
        self.kp = 50
        self.kd = 2.5
        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)

        # State variables
        self.q = np.zeros(12, dtype=np.float64)
        self.qr = np.zeros(12, dtype=np.float64)
        self.dq = np.zeros(12, dtype=np.float64)
        self.dqr = np.zeros(12, dtype=np.float64)
        self.delta_qr = np.zeros(12, dtype=np.float64)

        self.tau_max = np.array([23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43, 23.7, 23.7, 45.43])

        self.base_pos = np.zeros(3)
        self.base_orn = np.zeros(4)
        self.iterations = 0
        self.robot_states = RobotStates()

        # self.reset_robot_pose()

        config = {
            'model': self.pin_model,
            'data': self.pin_data,
            'geo_model': self.geo_model,
            'geo_data': self.geo_data,
            'links': self.links,
            'legs': self.legs,
            'links_ids': self.links_ids,
            'foot_ids': self.foot_ids,
            'kp': self.kp,
            'kd': self.kd,
            'robot_states': self.robot_states
        }

        # Unitree strategy
        # self.task_control = UnitreeSR(**config)
        self.task_control = ControlScheduler(**config)

        self.base_rgc = BaseRGC(**config)

        self._update_robot_sim_states()

        # SYNC VIEWER AFTER RESET
        if self._is_render and self.viewer:
            self.viewer.sync()

    def _setup_joint_mapping(self):
        """Set up joint name to index mapping with FR, FL, RR, RL order"""
        self.joint_names = []
        self.actuator_names = []

        for leg in self.legs:
            for link in self.links:
                joint_name = f"{leg}_{link}_joint"
                actuator_name = f"{leg}_{link}"

                self.joint_names.append(joint_name)
                self.actuator_names.append(actuator_name)

        # Get joint indices and qpos addresses
        self.joint_idx_list = []
        self.joint_qpos_addr = []  # Store the actual qpos addresses

        for name in self.joint_names:
            try:
                joint_id = self.model.joint(name).id
                self.joint_idx_list.append(joint_id)
                # Get the qpos address for this joint
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.joint_qpos_addr.append(qpos_addr)
            except Exception as e:
                print(f"Error with joint {name}: {e}")
                # Add placeholder values to maintain order
                self.joint_idx_list.append(-1)
                self.joint_qpos_addr.append(-1)

        # Get actuator indices
        self.actuator_idx_list = []
        for name in self.actuator_names:
            try:
                actuator_id = self.model.actuator(name).id
                self.actuator_idx_list.append(actuator_id)
            except Exception as e:
                print(f"Error with actuator {name}: {e}")
                self.actuator_idx_list.append(-1)

        # Setup pinocchio frame IDs if pinocchio model is available
        if self.pin_model is not None:
            for leg in self.legs:
                for l_name in self.links:
                    frame_name = f'{leg}_{l_name}'
                    try:
                        frame_id = self.pin_model.getFrameId(frame_name)
                        self.links_ids.append(frame_id)
                    except:
                        print(f"Warning: Frame {frame_name} not found in Pinocchio model")

                foot_name = f'{leg}_foot'
                try:
                    foot_id = self.pin_model.getFrameId(foot_name)
                    self.foot_ids.append(foot_id)
                except:
                    print(f"Warning: Frame {foot_name} not found in Pinocchio model")

    def _physics(self, tau):
        """Apply torques and step physics"""
        # Only apply torques to valid actuators
        valid_tau = []
        for i, actuator_idx in enumerate(self.actuator_idx_list):
            if actuator_idx != -1:
                self.data.ctrl[actuator_idx] = tau[i]
            else:
                print(f"Warning: Invalid actuator index for torque application")

        mujoco.mj_step(self.model, self.data)

    def _low_level_control(self):
        """Compute low-level PD control"""
        self._update_robot_sim_states()
        tau = self.KP @ (self.qr - self.q) + self.KD @ (self.dqr - self.dq)
        self.robot_states.tau_pd = tau.reshape(12, 1)

        if self.pin_model is not None:
            tau_g = self._comp_tau_g()
        else:
            tau_g = np.zeros(12)

        return np.clip(tau + tau_g, -self.tau_max, self.tau_max)

    def _comp_tau_g(self):
        """Compute gravity compensation torques using Pinocchio"""
        if self.pin_model is None:
            return np.zeros(12)

        q_full = np.vstack((self.robot_states.b_pos, self.robot_states.epsilon, self.robot_states.q[3:6],
                            self.robot_states.q[0:3], self.robot_states.q[9:12], self.robot_states.q[6:9]))
        tau_g = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q_full)[6:]
        tau_g = np.vstack((tau_g[3:6], tau_g[0:3], tau_g[9:12], tau_g[6:9])).reshape(12)
        self.robot_states.tau_g = tau_g.reshape(12, 1)
        return tau_g

    def _task_control(self, mode=None):
        """Execute task-level control"""
        self.qr += self.delta_qr
        self.robot_states.qr = self.qr.reshape(12, 1)
        self.delta_qr, self.KP, self.KD = self.task_control.update(mode)

    def _update_robot_sim_states(self):
        """Update robot state variables from simulation"""
        # Joint positions and velocities using qpos addresses
        for i, qpos_addr in enumerate(self.joint_qpos_addr):
            if qpos_addr != -1 and qpos_addr < self.model.nq:
                self.q[i] = self.data.qpos[qpos_addr]
                if self.joint_idx_list[i] != -1:
                    qvel_addr = self.model.jnt_dofadr[self.joint_idx_list[i]]
                    if qvel_addr < self.model.nv:
                        self.dq[i] = self.data.qvel[qvel_addr]

        # Base pose and velocity
        self.base_pos = self.data.qpos[0:3]
        self.base_orn = np.array((self.data.qpos[4], self.data.qpos[5], self.data.qpos[6], self.data.qpos[3]))
        self.base_lin_vel = self.data.qvel[0:3]
        self.base_ang_vel = self.data.qvel[3:6]

        # Update robot states object
        self.robot_states.b_pos = self.base_pos.reshape(3, 1)
        self.robot_states.epsilon = self.base_orn.reshape(4, 1)
        self.robot_states.b_vel = self.base_lin_vel.reshape(3, 1)
        self.robot_states.omega = self.base_ang_vel.reshape(3, 1)
        self.robot_states.rpy = self.quaternion_to_rpy(self.robot_states.epsilon).reshape(3, 1)
        self.robot_states.q = self.q.reshape(12, 1)
        self.robot_states.dq = self.dq.reshape(12, 1)

        self.base_rgc.com_quatities()

    def control_loop(self, mode):
        """Main control loop"""
        if mode != -1:
            self._task_control(mode=mode)

        for _ in range(int(self.con_dt / self.dyn_dt)):
            tau = self._low_level_control()
            self._physics(tau=tau)

        if self._is_render and self.viewer:
            # if self.task_control.using_mpc:
            #     self.visualize_simple_hull()
            self.viewer.sync()
            time.sleep(self.con_dt)

        self.iterations += 1

    def reset_robot_pose(self, q0=None, b0=None, r0=None):
        """Reset robot to initial pose"""
        if q0 is None:
            # Default joint positions in FR, FL, RR, RL order
            # random start pose
            # q0 = [0.9, 2, -1.65, -0.6, 1.86, -1.65, -0.5, 1.06, -1.0, 0.25, 1.36, -1.05]

            # safe pose
            q0 = [0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7]

            # q0 = [0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7]  # upside

            # Side way
            # q0 = [-0.25, 0.90, -2.85, -0.85, 0.85, -1.3, -0.25, 0.90, -2.85, 0.6, 3.75, -1.5]

        if b0 is None:
            b0 = [0, 0, 0.085]
            # b0 = [0, 0, 0.17]
            # b0 = [0, 0, 0.7]
        if r0 is None:
            r0 = [np.pi, 0, 0]
            # r0 = [np.pi * 100 / 180, 0, 0]
            # r0 = [0, 0, 0]

        # print(f"Resetting robot to position: {b0}, orientation: {r0}")

        # Reset base position and orientation
        self.data.qpos[0:3] = b0
        quat = self._euler_to_quat(r0)
        self.data.qpos[3:7] = quat

        # Reset joint positions using qpos addresses
        for i, qpos_addr in enumerate(self.joint_qpos_addr):
            if qpos_addr != -1 and qpos_addr < self.model.nq:
                self.data.qpos[qpos_addr] = q0[i]
                # print(f"Setting joint {self.joint_names[i]} (qpos_addr {qpos_addr}) to {q0[i]}")

        # Reset velocities to zero
        self.data.qvel[:] = 0
        self.qr = np.array(q0).reshape(12)

        # Forward dynamics to update the simulation
        mujoco.mj_forward(self.model, self.data)

        render_aux = self._is_render
        self._is_render = False
        for _ in range(100):
            self.control_loop(-1)
        self._is_render = render_aux

        # print("Reset complete")

    def _euler_to_quat(self, euler):
        """Convert Euler angles to quaternion in MuJoCo's [w, x, y, z] order"""

        r = Rotation.from_euler('xyz', euler)
        quat_xyzw = r.as_quat()

        quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

        return quat_wxyz

    def quaternion_to_rpy(self, quaternion):
        """        
        Parameters:
        quaternion: numpy array of shape (4, 1) with [[x], [y], [z], [w]] components
        
        Returns:
        roll, pitch, yaw: Euler angles in radians
        """
        x = quaternion[0, 0]
        y = quaternion[1, 0]
        z = quaternion[2, 0]
        w = quaternion[3, 0]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            # Use 90 degrees if out of range
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def close(self):
        """Close the simulation"""
        if self.viewer:
            self.viewer.close()

    def visualize_simple_hull(self):
        """
        Correct convex hull visualization for newer MuJoCo Python bindings
        """
        contacts = self.robot_states.contacts[0:3]
        com_pos = self.robot_states.r_pos
        contacts_2d = np.array([contact[:2] for contact in contacts])

        from scipy.spatial import ConvexHull
        hull = ConvexHull(contacts_2d)
        vertices_2d = contacts_2d[hull.vertices]

        z_offset = 0.01

        self.viewer.user_scn.ngeom = 0

        for i, vertex in enumerate(vertices_2d):
            pos = [vertex[0], vertex[1], z_offset]

            geom_id = self.viewer.user_scn.ngeom
            self.viewer.user_scn.ngeom += 1

            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[geom_id],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0.01, 0.01],  # Radius
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=[1, 0, 0, 1]  # Red
            )

        for i in range(len(vertices_2d)):
            v1 = vertices_2d[i]
            v2 = vertices_2d[(i + 1) % len(vertices_2d)]
            start = np.array([v1[0], v1[1], z_offset])
            end = np.array([v2[0], v2[1], z_offset])

            geom_id = self.viewer.user_scn.ngeom
            self.viewer.user_scn.ngeom += 1
            midpoint = (start + end) / 2
            full_length = np.linalg.norm(end - start)
            half_length = full_length / 2

            mujoco.mjv_initGeom(self.viewer.user_scn.geoms[geom_id],
                                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.001, half_length, 0.001],
                                pos=midpoint,
                                mat=self.vector_rotation(start, end),
                                rgba=[0, 1, 0, 1])

            edge_vec = v2 - v1
            normal = np.array([edge_vec[1], -edge_vec[0]])  # Rotate 90° clockwise
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-10:
                normal_unit = normal / normal_norm
            else:
                normal_unit = normal

            centroid = np.mean(vertices_2d, axis=0)
            if np.dot(normal_unit, v1 - centroid) < 0:
                normal_unit = -normal_unit

            normal_scale = 0.05
            normal_start_2d = (v1 + v2) / 2
            normal_end_2d = normal_start_2d + normal_unit * normal_scale

            normal_start_3d = np.array([normal_start_2d[0], normal_start_2d[1], z_offset + 0.005])
            normal_end_3d = np.array([normal_end_2d[0], normal_end_2d[1], z_offset + 0.005])

            arrow_length = np.linalg.norm(normal_end_3d - normal_start_3d)

            geom_id = self.viewer.user_scn.ngeom
            self.viewer.user_scn.ngeom += 1
            mujoco.mjv_initGeom(self.viewer.user_scn.geoms[geom_id],
                                type=mujoco.mjtGeom.mjGEOM_ARROW,
                                size=[0.002, 0.002, arrow_length],
                                pos=normal_start_3d,
                                mat=self.vector_rotation(normal_start_3d, normal_end_3d),
                                rgba=[0, 0, 1, 1])

        # Visualize CoM as HUGE sphere
        geom_id = self.viewer.user_scn.ngeom
        self.viewer.user_scn.ngeom += 1
        mujoco.mjv_initGeom(self.viewer.user_scn.geoms[geom_id],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.01, 0.01, 0.051],
                            pos=[com_pos[0, 0], com_pos[1, 0], 0],
                            mat=np.eye(3).flatten(),
                            rgba=[1, 1, 0, 1])

    def vector_rotation(self, start, end):
        """Create rotation matrix to align capsule with line direction"""
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return np.eye(3).flatten()
        direction = direction / length
        # Default capsule is along Z-axis, we want to align with our direction
        z_axis = np.array([0, 0, 1])
        # Avoid issues with parallel vectors
        if np.abs(np.dot(direction, z_axis)) > 0.99:
            y_axis = np.array([0, 1, 0])
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = np.cross(z_axis, direction)
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(direction, x_axis)
        # Create rotation matrix
        rot_mat = np.column_stack([x_axis, y_axis, direction])
        return rot_mat.flatten()

    def setup_collision_pairs(self):
        leg_geoms = ["FL_calf_0", "FR_calf_0", "RL_calf_0", "RR_calf_0"]

        pairs = list(itertools.combinations(leg_geoms, 2))

        for name_A, name_B in pairs:
            try:
                id_A = self.geo_model.getGeometryId(name_A)
                id_B = self.geo_model.getGeometryId(name_B)

                # Create and add the pair
                self.geo_model.addCollisionPair(pin.CollisionPair(id_A, id_B))

            except KeyError:
                print(f"WARNING: Could not find {name_A} or {name_B}")

        self.geo_data = pin.GeometryData(self.geo_model)


# Usage example
if __name__ == "__main__":
    sim = Go2ModelSimMuJoCo(render=True)
    q0 = [0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7]
    b0 = [2, 0, 0.8]
    r0 = [0, 0, 0.25]
    sim.reset_robot_pose(q0=q0, b0=b0, r0=r0)
    time.sleep(0.2)
    try:
        tick = 0
        while (True):
            if tick < 100:
                mode = -1
            else:
                mode = 4
            sim.control_loop(mode=mode)
            tick += 1
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        sim.close()
