import os
import json
import torch
import numpy as np
import time
import mujoco
import mujoco.viewer

from env.go2_env import Go2Env
from env.normalizer import StateNormalizer
from env.tasks.self_righting_task import SelfRightingTask
from tpe.tpe_module import TPEModule
from control.self_righting.time_based_solution.time_based_scheduler import SchedulerTB

MAX_INT = 3000


def main():

    model = mujoco.MjModel.from_xml_path("sim/assets/unitree_go2/scene.xml")
    data = mujoco.MjData(model)

    controller = SchedulerTB()
    normalizer = StateNormalizer(joint_limits=1, torque_limits=1)
    tpe = TPEModule(model_path="tpe_model.pt")

    task = SelfRightingTask(normalizer=normalizer, tpe=tpe)

    viewer = mujoco.viewer.launch_passive(model, data)
    config = {
        "urdf_path": "sim/assets/unitree_go2/go2.urdf",
        "mj_model": model,
        "mj_data": data,
        "controller": controller,
        "task": task,
        "viewer": viewer
    }

    env = Go2Env(max_step=MAX_INT, **config)
    action = 2
    tick = 0
    total_reward = 0
    env.reset()
    end_sim = False
    while not end_sim:
        action = dummy_rule(tick)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        end_sim = terminated or truncated
        tick += 1


def dummy_rule(tick):
    if tick < 25:
        action = 0
    elif tick < 25 + 80:
        action = 1
    elif tick < 25 + 80 + 310:
        action = 2
    elif tick < 25 + 80 + 310 + 370:
        action = 3
    elif tick < 25 + 80 + 310 + 370 + 300:
        action = 4
    elif tick < 25 + 80 + 310 + 370 + 300 + 210:
        action = 5
    elif tick < 25 + 80 + 310 + 370 + 300 + 210 + 210:
        action = 6
    else:
        action = 0

    return action


if __name__ == "__main__":
    main()
