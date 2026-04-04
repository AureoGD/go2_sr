import mujoco
from env.go2_env import Go2Env
from control.self_righting.time_based_solution.time_based_scheduler import SchedulerTB
from env.normalizer import StateNormalizer
from tpe.tpe_module import TPEModule
from env.tasks.self_righting_task import SelfRightingTask


def create_env(rendering=False):

    # -----------------------------
    # Mujoco
    # -----------------------------
    model = mujoco.MjModel.from_xml_path("sim/assets/unitree_go2/scene.xml")
    data = mujoco.MjData(model)

    viewer = None
    if rendering:
        viewer = mujoco.viewer.launch_passive(model, data)

    # -----------------------------
    # Components
    # -----------------------------
    controller = SchedulerTB()

    normalizer = StateNormalizer(joint_limits=1, torque_limits=1)

    tpe = TPEModule(model_path="tpe_model.pt")

    task = SelfRightingTask(normalizer=normalizer, tpe=tpe)

    # -----------------------------
    # Env
    # -----------------------------
    config = {
        "urdf_path": "sim/assets/unitree_go2/go2.urdf",
        "mj_model": model,
        "mj_data": data,
        "controller": controller,
        "task": task,
        "viewer": viewer
    }

    env = Go2Env(max_step=100, **config)

    return env
