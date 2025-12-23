import multiprocessing as mp
# Ensure 'spawn' is used for safety with MuJoCo/PyTorch
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

import os
import torch
import numpy as np
from typing import Dict, Any, Tuple
from types import SimpleNamespace

from es_framework.components.nn_utils import unflatten_nn_parameters
from es_framework.components.policy import Policy
from environment.go2_env import Go2Env

# --- Global Worker Variables ---
# These exist once per CPU core to save memory
sim_instance = None
policy_net = None


def init_worker(config: Dict[str, Any]):
    """
    Initialize the worker process. 
    This runs ONCE when the pool starts.
    """
    global sim_instance, policy_net

    # 1. Performance Optimization: Force Single-Threaded Math
    # Since we run many workers in parallel, we don't want them fighting for threads.
    import torch

    torch.set_num_threads(1)

    process_id = os.getpid()

    # 2. Setup Environment & Network
    # We look for 'policy_config' or 'model_config' in the dictionary
    policy_cfg = config.get('model_config', {})

    # Create the environment (Headless)
    sim_instance = Go2Env(env_id=process_id, rendering=False)

    # Create the Policy Structure
    output_dim = sim_instance.action_space.n if hasattr(sim_instance.action_space,
                                                        'n') else sim_instance.action_space.shape

    # Instantiate the Policy (formerly ControlRule)
    policy_net = Policy(observation_dim=sim_instance.observation_space.shape[0], output_dim=output_dim, **policy_cfg)
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_micro_task(task_args: Tuple):
    """
    Executes a SINGLE simulation scenario.
    
    Args:
        task_args: Tuple containing:
            - ind_id (int): ID of the individual (0-19)
            - nn_params_flat (np.array): Weights for the Neural Net
            - single_condition (list): [q0, r0, b0] for reset
            - difficulty (float): Current curriculum difficulty
            - norm_stats (dict or Obj): Frozen normalization stats
            
    Returns:
        (ind_id, episode_reward, new_stats, max_stage_reached)
    """
    global sim_instance, policy_net

    # Unpack the single micro-task
    ind_id, nn_params_flat, single_condition, difficulty, norm_stats = task_args

    # 1. Configure Environment Difficulty
    # (Assuming sim_instance has a method/property to set difficulty if using Curriculum)
    if hasattr(sim_instance, 'set_difficulty'):
        sim_instance.set_difficulty(difficulty)
    elif hasattr(sim_instance, 'tasks'):
        sim_instance.tasks.set_difficulty(difficulty)

    try:
        # 2. Load Weights into Policy
        state_dict = unflatten_nn_parameters(nn_params_flat, policy_net)
        policy_net.load_state_dict(state_dict)

        # 3. Sync Normalizer (Apply the "Frozen Ruler")
        # Handle both Dictionary and Object cases for robustness
        if norm_stats is not None:
            if isinstance(norm_stats, dict):
                norm_stats_obj = SimpleNamespace(**norm_stats)
                sim_instance.normalizer.sync_global_stats(norm_stats_obj)
            else:
                sim_instance.normalizer.sync_global_stats(norm_stats)
        else:
            # Fallback for very first generation
            sim_instance.normalizer.reset_shadow()

        # 4. Run ONE Episode
        max_stage_reached = -1
        episode_reward = 0.0

        # Reset with the specific condition
        # Ensure your Go2Env.reset accepts these args!
        q0, r0, b0 = single_condition
        obs, _ = sim_instance.reset(q0=q0, r0=r0, b0=b0)

        # Step Loop
        # Use a hard limit or the env's internal limit
        max_steps = getattr(sim_instance, 'max_step_limit', 1000)

        for _ in range(max_steps):

            # Action (No Grad needed for inference)
            with torch.no_grad():
                action, _ = policy_net.predict(obs)

            # Step
            obs, reward, terminated, truncated, _ = sim_instance.step(action)

            # Crash Check (Safety - depends on your Sim implementation)
            # If you moved critical check to Env, check 'terminated' directly
            episode_reward += reward

            if terminated or truncated:
                break

        # 5. Check Curriculum Progress
        # (Accessing the inner sim state to see if "Stand" or "Flip" happened)
        # if hasattr(sim_instance, 'sim') and hasattr(sim_instance.sim, 'robot_states'):
        #     completed_dict = sim_instance.sim.robot_states.sr_mode_completed
        #     for mode_idx, is_done in completed_dict.items():
        #         if is_done:
        #             max_stage_reached = max(max_stage_reached, mode_idx)

        # 6. Get New Statistics (Shadow Mode)
        new_stats = sim_instance.normalizer.get_shadow_stats()

        return ind_id, episode_reward, new_stats, max_stage_reached

    except Exception as e:
        # Robustness: If one sim crashes, print why but don't kill training
        print(f"[Worker Error] Ind {ind_id}: {e}")
        import traceback
        traceback.print_exc()
        return ind_id, -1000.0, None, -1.0
