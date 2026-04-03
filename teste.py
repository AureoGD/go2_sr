import numpy as np
import mujoco
import mujoco.viewer
import time

from sim.go2_sim import Go2Sim
from control.base_controller import BaseController


# ======================================================
# SIMPLE TEST CONTROLLER
# ======================================================
class JointTestController(BaseController):

    def __init__(self, joint_id=0, amplitude=0.5, frequency=0.5):
        super().__init__()

        self.joint_id = joint_id
        self.amp = amplitude
        self.freq = frequency
        self.t = 0.0

    def before_step(self, state, action):

        q = state.robot.q.copy()

        # sinal senoidal em UMA junta
        q[self.joint_id] += self.amp * np.sin(2 * np.pi * self.freq * self.t)

        state.robot.qr = q

        # ganhos constantes
        state.controller.Kp = np.ones(12) * 50.0
        state.controller.Kd = np.ones(12) * 2.0

        self.t += 0.01

    def after_step(self, state):
        pass

    def get_action_space(self):
        return super().get_action_space()


# ======================================================
# MAIN TEST
# ======================================================
def main():

    # -------------------------------
    # LOAD MUJOCO MODEL
    # -------------------------------
    model = mujoco.MjModel.from_xml_path("sim/assets/unitree_go2/scene.xml")
    data = mujoco.MjData(model)

    # -------------------------------
    # VIEWER
    # -------------------------------
    viewer = mujoco.viewer.launch_passive(model, data)

    # -------------------------------
    # CONTROLLER
    # -------------------------------
    controller = JointTestController(joint_id=0)

    # -------------------------------
    # SIMULATION
    # -------------------------------
    sim = Go2Sim(urdf_path="sim/assets/unitree_go2/go2.urdf",
                 mj_model=model,
                 mj_data=data,
                 controller=controller,
                 viewer=viewer)

    sim.reset_robot_pose(r0=[np.pi, 0, 0])

    while viewer.is_running():

        sim.simulation_loop(action=None)

        time.sleep(sim.con_dt)


if __name__ == "__main__":
    main()
