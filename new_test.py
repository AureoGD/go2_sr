from env.go2_env import Go2Env
from control.self_righting.time_based_solution.time_based_scheduler import SchedulerTB
from env.tasks.self_righting_task import SelfRightingTask
from env.env_factory import create_env
from es_framework.core.env_config import EnvConfig
from env.tasks.self_righting_scenario import SelfRightingScenario
from guided_mutation.es_framework.models.policy import Policy
import numpy as np
import time
import random

import pickle
import os

with open("/home/CCT/9791086/mujoco_sr/failed_tasks/failed_task_3.pkl", "rb") as f:
    data = pickle.load(f)

params = data[3]
options = data[4]

config = {

    # -----------------------------
    # Experiment
    # -----------------------------
    "job_name": "go2_self_righting",

    # -----------------------------
    # Optimizer
    # -----------------------------
    "optimizer_type": "CEM",

    # -----------------------------
    # Population Training
    # -----------------------------
    "pop_size": 5,
    "species_size": 1,
    "num_scenarios": 5,
    "max_generations": 1000,
    "max_workers": 15,
    "max_steps": 1000,

    # -----------------------------
    # ES params
    # -----------------------------
    "sigma_init": 0.05,
    "sigma_decay": 0.995,
    "elite_frac": 0.2,

    # -----------------------------
    # RL params
    # -----------------------------
    "rl_steps": 1000,
    "batch_size": 256,
    "gamma": 0.99,
    "epsilon": 0.2,

    # -----------------------------
    # V-guided exploration
    # -----------------------------
    "delta_v": 20.0,
    "epsilon_boost": 1.3,
    "epsilon_max": 0.6,
    "v_policy": {
        "window_size": 10,
        "batch_size": 256
    },

    # -----------------------------
    # Model
    # -----------------------------
    "model_config": {
        "layers": [
            {
                "units": 64,
                "activation": "tanh"
            },
            {
                "units": 64,
                "activation": "tanh"
            },
        ]
    },

    # -----------------------------
    # Difficulty (task-level)
    # -----------------------------
    "difficulty": 0,
    "env_config": None,
    "scenario_generator_class": None,
}

env_config = EnvConfig(env_class=Go2Env,
                       controller_class=SchedulerTB,
                       task_class=SelfRightingTask,
                       scene_path="sim/assets/unitree_go2/scene.xml",
                       urdf_path="sim/assets/unitree_go2/go2.urdf",
                       tpe_model_path="tpe/models/tpe_cnn/best_model.pt",
                       normalizer_params={
                           "joint_limits": 1,
                           "torque_limits": 1
                       },
                       render=True)


def main(_env_config):

    env, env_spec = create_env(_env_config)
    scene = SelfRightingScenario()
    policy = Policy(env_spec, config["model_config"])
    policy.set_parameters(params)

    for i in range(100):
        tick = 0
        total_reward = 0
        # options = scene.sample()
        # options = {
        #     'q0': np.array([-0., 1.41, -2.72, -0.02, 1.4, -2.72, -0.02, 1.44, -2.73, -0.01, 1.39, -2.73]),
        #     'r0': np.array([0.01, -0.02, 1.7473776]),
        #     'b0': np.array([-0.99161911, -0.76641263, 0.18])
        # }
        st, _ = env.reset(options=options)
        end_sim = False
        time_now = time.time()
        while not end_sim:

            action, _ = policy.predict(st)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            end_sim = terminated or truncated
            tick += 1
            st = obs
            if (tick % 200 == 0):
                print("Doing something")

        print(f"Scenario: {i+1} - Tick: {tick}")

    del policy

    policy = Policy(env_spec, config["model_config"])


def dummy_rule(tick):
    action = 0

    if tick < 25:
        action = 0
    elif tick < 25 + 210:
        action = 5
    elif tick < 25 + 210 + 310:
        action = 6
    # elif tick < 25 + 80 + 310 + 370:
    #     action = 8
    # elif tick < 25 + 80 + 310 + 370 + 300:
    #     action = 9
    # elif tick < 25 + 80 + 310 + 370 + 300 + 210:
    #     action = 10
    # elif tick < 25 + 80 + 310 + 370 + 300 + 210 + 210:
    #     action = 6
    # else:
    #     action = 0

    return action


def debug_rgc(tick):

    if tick < 20:
        action = 0
    elif tick < 150:
        action = 1  # go_safe
    elif tick < 150 + 250:
        action = 2  # prepare_cw
    elif tick < 150 + 250 + 150:
        action = 3  # roll_cw
    elif tick < 150 + 250 + 150 + 250:
        action = 4  # landing_cw
    elif tick < 150 + 250 + 150 + 250 + 270:
        action = 5  # prone
    elif tick < 150 + 250 + 150 + 250 + 270 + 300:
        action = 6  # standing_up
    else:
        action = 0


if __name__ == "__main__":
    main(env_config)
