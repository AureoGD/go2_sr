import os
import time
import signal
import torch
from typing import Dict, Any, Tuple
from types import SimpleNamespace

from es_framework.components.nn_utils import unflatten_nn_parameters
from es_framework.components.policy import Policy
from env.go2_env import Go2Env


def init_worker(config, heartbeat_array):
    global env, policy_net, shared_heartbeat, worker_model_config

    torch.set_num_threads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    shared_heartbeat = heartbeat_array
    worker_model_config = config["model_config"]

    env = Go2Env(rendering=False)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = Policy(obs_dim, act_dim, **worker_model_config)


def run_micro_task(task_args):
    global env, policy_net, shared_heartbeat

    ind_id, nn_params_flat, single_condition, difficulty, norm_stats = task_args
    shared_heartbeat[ind_id] = time.time()

    # Load policy weights
    state_dict = unflatten_nn_parameters(nn_params_flat, policy_net)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()

    # Configure env
    env.set_difficulty(difficulty)

    q0, r0, b0, mode = single_condition
    obs, _ = env.reset(q0=q0, r0=r0, b0=b0, mode=mode)

    total_reward = 0.0
    max_steps = getattr(env, "max_step_limit", 1000)

    for step in range(max_steps):
        if step % 25 == 0:
            shared_heartbeat[ind_id] = time.time()

        with torch.no_grad():
            action, _ = policy_net.predict(obs)

        obs, r, terminated, truncated, _ = env.step(int(action))
        total_reward += r

        if terminated or truncated:
            break

    shared_heartbeat[ind_id] = -1.0
    return ind_id, total_reward, -1
