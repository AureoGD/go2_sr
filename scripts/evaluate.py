import argparse
import json
import torch
import numpy as np

from env.env_factory import create_env
from es_framework.models.policy import Policy
from es_framework.core.env_spec import EnvSpec
from es_framework.core.env_config import EnvConfig

# ----------------------------------------
# IMPORTS PARA MAPEAR CLASSES
# ----------------------------------------
from env.go2_env import Go2Env
from control.self_righting.time_based_solution.time_based_scheduler import SchedulerTB
from env.tasks.self_righting_task import SelfRightingTask

# ----------------------------------------
# CLASS MAP (STRING → CLASSE)
# ----------------------------------------
CLASS_MAP = {
    "Go2Env": Go2Env,
    "SchedulerTB": SchedulerTB,
    "SelfRightingTask": SelfRightingTask,
}

# ----------------------------------------
# ARGUMENTS
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--model", type=str, default="best")  # best | last | gen_x
args = parser.parse_args()

RUN_DIR = args.run_dir

# ----------------------------------------
# PATHS
# ----------------------------------------
CONFIG_PATH = f"{RUN_DIR}/config.json"
MODEL_DIR = f"{RUN_DIR}/models"
SPEC_PATH = f"{MODEL_DIR}/model_spec.pt"

if args.model == "best":
    PARAMS_PATH = f"{MODEL_DIR}/best_params.npy"
elif args.model == "last":
    PARAMS_PATH = f"{MODEL_DIR}/last_params.npy"
else:
    PARAMS_PATH = f"{MODEL_DIR}/checkpoints/{args.model}.npy"

# ----------------------------------------
# LOAD CONFIG
# ----------------------------------------
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

env_cfg_dict = config["env_config"]

# ----------------------------------------
# RECONSTRUIR ENV CONFIG
# ----------------------------------------
env_config = EnvConfig(
    env_class=CLASS_MAP[env_cfg_dict["env_class"]],
    controller_class=CLASS_MAP[env_cfg_dict["controller_class"]],
    task_class=CLASS_MAP[env_cfg_dict["task_class"]],
    scene_path=env_cfg_dict["scene_path"],
    urdf_path=env_cfg_dict["urdf_path"],
    tpe_model_path=env_cfg_dict["tpe_model_path"],
    normalizer_params=env_cfg_dict["normalizer_params"],
    render=True  # 🔥 diferença principal
)

# ----------------------------------------
# LOAD MODEL SPEC
# ----------------------------------------
spec = torch.load(SPEC_PATH, map_location="cpu", weights_only=False)

# Caso tenha salvo como dict puro
if isinstance(spec["env_spec"], dict):
    env_spec = EnvSpec(**spec["env_spec"])
else:
    env_spec = spec["env_spec"]

model_cfg = spec["model_cfg"]

# ----------------------------------------
# BUILD POLICY
# ----------------------------------------
policy = Policy(env_spec, model_cfg)

# ----------------------------------------
# LOAD PARAMS (ES STYLE)
# ----------------------------------------
params = np.load(PARAMS_PATH)
policy.set_parameters(params)

policy.eval()

print(f"[INFO] Loaded model from: {PARAMS_PATH}")

# ----------------------------------------
# CREATE ENV
# ----------------------------------------
env, _ = create_env(env_config)

# ----------------------------------------
# RUN EPISODE
# ----------------------------------------
obs, _ = env.reset()

done = False
total_reward = 0.0
step = 0

while not done:

    with torch.no_grad():
        action, _ = policy.predict(obs)

    # se ação for discreta
    if env_spec.is_discrete:
        action = int(action)

    obs, reward, terminated, truncated, _ = env.step(action)

    total_reward += reward
    step += 1

    done = terminated or truncated

print("\n----------------------------------")
print(f"Episode finished")
print(f"Steps:   {step}")
print(f"Reward:  {total_reward:.3f}")
print("----------------------------------\n")

env.close()
