import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import multiprocessing as mp

from es_framework.core.trainer import Trainer
from es_framework.core.env_config import EnvConfig
from env.env_factory import create_env
from es_framework.models.policy import Policy

# ----------------------------------------
# Imports específicos do projeto
# ----------------------------------------
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
    "optimizer_type": "CEM",
    "job_name": "go2_self_righting",

    # -----------------------------
    # Training
    # -----------------------------
    "pop_size": 40,
    "num_scenarios": 5,
    "max_generations": 1000,
    "max_workers": 20,

    # -----------------------------
    # ES params
    # -----------------------------
    "sigma_init": 0.05,
    "sigma_decay": 0.995,
    "elite_frac": 0.2,

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
    "difficulty": 0
}

# ----------------------------------------
# ENV CONFIG
# ----------------------------------------
env_config = EnvConfig(env_class=Go2Env,
                       controller_class=SchedulerTB,
                       task_class=SelfRightingTask,
                       scene_path="sim/assets/unitree_go2/scene.xml",
                       urdf_path="sim/assets/unitree_go2/go2.urdf",
                       tpe_model_path="tpe_model.pt",
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

    print(f"[INIT] Number of parameters: {num_params}")

    trainer = Trainer(config)
    trainer.train()
