import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ----------------------------------------
# IMPORTS
# ----------------------------------------
# General
import numpy as np
import multiprocessing as mp

# Framework
from guided_mutation.es_framework.core.trainer import Trainer
from guided_mutation.es_framework.models.policy import Policy
from es_framework.core.env_config import EnvConfig
from env.env_factory import create_env

# Env
from env.go2_env import Go2Env
from control.self_righting.time_based_solution.time_based_scheduler import SchedulerTB
from env.tasks.self_righting_task import SelfRightingTask
from env.tasks.self_righting_scenario import SelfRightingScenario

# ----------------------------------------
# CONFIG
# ----------------------------------------
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
    "pop_size": 20,
    "species_size": 1,
    "num_scenarios": 5,
    "max_generations": 1000,
    "max_workers": 20,
    "max_steps": 1000,

    # -----------------------------
    # ES params
    # -----------------------------
    "sigma_init": 0.2,
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

# ----------------------------------------
# ENV CONFIG
# ----------------------------------------
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
                       render=False)

config["env_config"] = env_config

# ----------------------------------------
# SCENARIO GENERATOR
# ----------------------------------------
config["scenario_generator_class"] = SelfRightingScenario

# ----------------------------------------
# SUCCESS CRITERION
# ----------------------------------------


def success_fcn(info, reward):
    return int(info.get("success_flag", False))


config["success_criterion"] = success_fcn

# ----------------------------------------
# INFER NUM PARAMS
# ----------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print("[INIT] Building dummy env to infer dimensions...")

    env, env_spec = create_env(env_config)
    policy = Policy(env_spec, config["model_config"])

    num_params = policy.num_parameters()
    config["num_params"] = num_params
    config["env_spec"] = env_spec

    if hasattr(env, "close"):
        env.close()

    del env

    config["env_fn"] = create_env

    print(f"[INIT] Number of parameters: {num_params}")

    trainer = Trainer(config)
    trainer.train()
