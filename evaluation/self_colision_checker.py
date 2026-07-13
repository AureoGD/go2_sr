import mujoco
import numpy as np

class SelfCollisionChecker:
    def __init__(self, xml_path, mj_order=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.mj_order = [0,1,2,3,4,5,6,7,8,9,10,11]  # map: Unitree order -> MuJoCo qpos order

    def is_self_colliding(self, q_unitree, penetration_tol=0.0):
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = 1.0          # base 1 m up: no ground contacts possible
        self.data.qpos[3] = 1.0          # identity quaternion (w, x, y, z)
        self.data.qpos[7:] = np.asarray(q_unitree)[self.mj_order]
        mujoco.mj_forward(self.model, self.data)

        for i in range(self.data.ncon):
            if self.data.contact[i].dist < penetration_tol:
                return True
        return False



