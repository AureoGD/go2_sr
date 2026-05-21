import time
import signal
import torch

from env.env_factory import create_env
from es_framework.models.policy import Policy
from es_framework.models.nn_utils import load_flat_params_into_model

# Globals (mantidos para multiprocessing)
env = None
policy_net = None
shared_heartbeat = None


# ----------------------------------------
# Init worker (executado uma vez por processo)
# ----------------------------------------
def init_worker(config, heartbeat_array):
    global env, policy_net, shared_heartbeat

    torch.set_num_threads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    shared_heartbeat = heartbeat_array

    env_config = config["env_config"]
    model_cfg = config["model_config"]

    env, env_spec = create_env(env_config)

    policy_net = Policy(env_spec, model_cfg)


def run_micro_task(task_args):
    global env, policy_net, shared_heartbeat

    task_id, ind_id, flat_params, scenario, difficulty, _ = task_args

    shared_heartbeat[task_id] = time.time()

    load_flat_params_into_model(flat_params, policy_net)

    if hasattr(env.controller, "reset_phase"):
        env.controller.reset_phase()

    if hasattr(env.task, "tpe"):
        env.task.tpe.reset()

    if hasattr(env.task, "set_difficulty"):
        env.task.set_difficulty(difficulty)

    q0, r0, b0 = scenario["q0"], scenario["r0"], scenario["b0"]
    obs, _ = env.reset(q0=q0, r0=r0, b0=b0)

    total_reward = 0.0
    step = 0
    end_sim = False

    while not end_sim:

        if step % 25 == 0:
            shared_heartbeat[task_id] = time.time()

        with torch.no_grad():
            action, _ = policy_net.predict(obs)

        obs, r, terminated, truncated, info = env.step(action)

        total_reward += r
        end_sim = terminated or truncated
        step += 1

        # Fail-safe for now! using 100 only for testing!
        if step > 2100:
            # print(f"[WARN] Forced break (ind {ind_id})")
            break

    success_flag = float(info["success_flag"])

    shared_heartbeat[task_id] = -1.0

    return task_id, ind_id, total_reward, success_flag
