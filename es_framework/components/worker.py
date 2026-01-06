import os
import time
import signal
import torch
from typing import Dict, Any, Tuple
from types import SimpleNamespace

from es_framework.components.nn_utils import unflatten_nn_parameters
from es_framework.components.policy import Policy
from environment.go2_env import Go2Env


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

    if norm_stats is not None:
        env.normalizer.sync_global_stats(norm_stats)
    else:
        env.normalizer.reset_shadow()

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
    return ind_id, total_reward, env.normalizer.get_shadow_stats(), -1


# # ============================================================
# # GLOBAL WORKER STATE (PER PROCESS)
# # ============================================================
# sim_instance = None
# policy_net = None
# shared_heartbeat = None

# # ============================================================
# # WORKER INITIALIZATION
# # ============================================================
# def init_worker(config: Dict[str, Any], heartbeat_array):
#     """
#     Called ONCE per worker process.
#     """
#     global sim_instance, policy_net, shared_heartbeat

#     torch.set_num_threads(1)
#     signal.signal(signal.SIGINT, signal.SIG_IGN)

#     shared_heartbeat = heartbeat_array

#     # --------------------------------------------------------
#     # Create persistent environment
#     # --------------------------------------------------------
#     sim_instance = Go2Env(env_id=os.getpid(), rendering=False)

#     # Keep your original destructor safety
#     sim_instance.close = lambda: None
#     if hasattr(sim_instance, "robot_sim"):
#         if hasattr(sim_instance.robot_sim, "close"):
#             sim_instance.robot_sim.close = lambda: None
#         if hasattr(sim_instance.robot_sim, "__del__"):
#             sim_instance.robot_sim.__del__ = lambda: None

#     # --------------------------------------------------------
#     # Policy structure (weights loaded per task)
#     # --------------------------------------------------------
#     policy_cfg = config.get("model_config", {})
#     obs_dim = sim_instance.observation_space.shape[0]

#     if hasattr(sim_instance.action_space, "n"):
#         act_dim = sim_instance.action_space.n
#     else:
#         act_dim = sim_instance.action_space.shape[0]

#     policy_net = Policy(observation_dim=obs_dim, output_dim=act_dim, **policy_cfg)

# # ============================================================
# # TASK EXECUTION
# # ============================================================
# def run_micro_task(task_args: Tuple):
#     """
#     Executes ONE rollout.
#     Heartbeat is indexed by TASK ID (Pool-safe).
#     """
#     global sim_instance, policy_net, shared_heartbeat

#     ind_id, nn_params_flat, single_condition, difficulty, norm_stats = task_args

#     try:
#         # ----------------------------------------------------
#         # Mark task as running
#         # ----------------------------------------------------
#         shared_heartbeat[ind_id] = time.time()

#         # Difficulty
#         if hasattr(sim_instance, "set_difficulty"):
#             sim_instance.set_difficulty(difficulty)

#         # Load policy parameters
#         state_dict = unflatten_nn_parameters(nn_params_flat, policy_net)
#         policy_net.load_state_dict(state_dict)

#         # Normalization
#         if norm_stats is not None:
#             if isinstance(norm_stats, dict):
#                 sim_instance.normalizer.sync_global_stats(SimpleNamespace(**norm_stats))
#             else:
#                 sim_instance.normalizer.sync_global_stats(norm_stats)
#         else:
#             sim_instance.normalizer.reset_shadow()

#         # Reset environment
#         q0, r0, b0, mode = single_condition
#         obs, _ = sim_instance.reset(q0=q0, r0=r0, b0=b0, mode=mode)

#         episode_reward = 0.0
#         max_steps = getattr(sim_instance, "max_step_limit", 1000)

#         policy_net.eval()

#         # ----------------------------------------------------
#         # Rollout loop
#         # ----------------------------------------------------
#         for step in range(max_steps):
#             if step % 25 == 0:
#                 shared_heartbeat[ind_id] = time.time()

#             with torch.no_grad():
#                 action, _ = policy_net.predict(obs)

#             action = int(action)

#             # Baseline safe action
#             obs, reward, terminated, truncated, _ = sim_instance.step(action)
#             episode_reward += reward

#             if terminated or truncated:
#                 break

#         # ----------------------------------------------------
#         # Task finished
#         # ----------------------------------------------------
#         shared_heartbeat[ind_id] = -1.0

#         shadow_stats = sim_instance.normalizer.get_shadow_stats()

#         max_level_reached = -1
#         try:
#             robot_sim = getattr(sim_instance, "robot_sim", None)
#             robot_states = getattr(robot_sim, "robot_states", None)
#             phases = getattr(robot_states, "end_phase_count", None)
#             if phases is not None:
#                 for idx, cnt in enumerate(phases):
#                     if cnt > 0:
#                         max_level_reached = max(max_level_reached, idx)
#         except Exception:
#             pass

#         return int(ind_id), float(episode_reward), shadow_stats, int(max_level_reached)

#     except Exception as e:
#         print(f"[Worker] Task {ind_id} ERROR: {e}", flush=True)
#         shared_heartbeat[ind_id] = -1.0
#         return int(ind_id), -1000.0, None, -1
