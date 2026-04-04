import os
import time
import signal
import torch
from typing import Dict, Any, Tuple
from types import SimpleNamespace

from es_framework.components.nn_utils import unflatten_nn_parameters
from es_framework.components.policy import Policy
from env.env_factory import create_env


def init_worker(config, heartbeat_array):
    global env, policy_net, shared_heartbeat, worker_model_config

    torch.set_num_threads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    shared_heartbeat = heartbeat_array
    worker_model_config = config["model_config"]

    env = create_env(rendering=False)

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

    step = 0
    end_sim = False

    while not end_sim:

        if step % 25 == 0:
            shared_heartbeat[ind_id] = time.time()

        with torch.no_grad():
            action, _ = policy_net.predict(obs)

        obs, r, terminated, truncated, info = env.step(int(action))
        total_reward += r

        end_sim = terminated or truncated
        step += 1

        # fail-safe leve
        if step > 5000:
            break

    shared_heartbeat[ind_id] = -1.0
    return ind_id, total_reward, -1
