import multiprocessing as mp
# Ensure start method is set to 'spawn' for CUDA/Torch compatibility/safety
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

import os
import torch
import numpy as np
from typing import Dict, Any, Tuple

from es_framework.commons.nn_parameters import unflatten_nn_parameters
from es_framework.commons.control_rule import ControlRule
from environment.env_go2 import Go2Env

# Global variables for the worker process (persistence across tasks)
sim_instance = None
control_rule = None
is_discrete = True


def init_worker(config: Dict[str, Any]):
    """
    Initialize the worker process. 
    Sets up the environment, neural network, and threading constraints.
    """
    global sim_instance, control_rule, is_discrete

    # --- PERFORMANCE FIX: FORCE SINGLE THREADING ---
    # Prevents "Thread Over-Subscription" where 20 workers * 8 threads = 160 threads choke the CPU.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Force PyTorch to stop spawning internal threads
    torch.set_num_threads(1)

    # --- PERFORMANCE FIX: UNLOCK CPU AFFINITY ---
    # Sometimes Python processes inherit a restrictive mask (only allowed on 1 core).
    # This unlocks the process to use any available CPU core.
    try:
        if hasattr(os, 'sched_getaffinity'):
            all_cpus = list(range(os.cpu_count()))
            os.sched_setaffinity(0, all_cpus)
    except Exception:
        pass  # Ignore if OS doesn't support this (e.g. Mac/Windows)

    process_id = os.getpid()

    # Extract Config
    control_rule_cfg = config.get('model_config', {})
    is_discrete = control_rule_cfg.get('discrete', False)

    # Initialize Environment (Headless)
    # print(f"[Worker {process_id}] Initializing...")
    sim_instance = Go2Env(env_id=process_id, rendering=False)

    output_dim = sim_instance.action_space.n

    # Initialize Control Rule (Neural Network)
    control_rule = ControlRule(observation_dim=sim_instance.observation_space.shape[0],
                               output_dim=output_dim,
                               **control_rule_cfg).reset_parameters(seed=45 + process_id, last_layer_std=1)


def run_worker(task_args: Tuple[int, np.ndarray, list, float, Any]):
    """
    Executes one evaluation task (one individual from the population).
    Runs N scenarios (initial conditions) and aggregates fitness.
    """
    global sim_instance, control_rule

    if sim_instance is None:
        raise RuntimeError(f"Worker {os.getpid()} not initialized correctly.")

    # Unpack arguments
    task_id, nn_params_flat, initial_conditions, difficulty, norm_stats = task_args

    # Set Curriculum Difficulty
    sim_instance.set_difficulty(difficulty)

    try:
        # 1. Load Neural Network Weights
        state_dict = unflatten_nn_parameters(nn_params_flat, control_rule)
        control_rule.load_state_dict(state_dict)

        # 2. Sync Normalizer (Actor)
        if norm_stats is not None:
            sim_instance.normalizer.sync_global_stats(norm_stats)
        else:
            sim_instance.normalizer.reset_shadow()

        max_step = sim_instance.max_step_limit

        cumulative_raw_reward = 0
        stages_reached = []

        # --- FITNESS CONFIGURATION ---
        # Weight for the "Average Progress" component.
        # This creates a gradient: Stage 0 is better than Nothing, Stage 1 is better than 0.
        TASK_PROGRESS_WEIGHT = 50.0
        CRITICAL_FAIL_PENALTY = -1000.0

        # --- EVALUATION LOOP (10 Scenarios) ---
        for initial_condition in initial_conditions:
            episode_reward = 0

            # Start at -1 to represent "No Tasks Completed".
            # If they finish Task 0 (Safe Mode), this becomes 0.
            current_run_max_stage = -1

            # Reset Environment
            obs, info = sim_instance.reset(q0=initial_condition[0], r0=initial_condition[1], b0=initial_condition[2])

            # Simulation Loop
            for step in range(max_step):
                # Predict
                action, _ = control_rule.predict(obs)

                # Step
                obs, reward, terminated, truncated, info = sim_instance.step(action)

                # --- SAFETY CHECK: MPC CRASH ---
                # Check directly in the robot state if the controller exploded
                if sim_instance.robot_sim.robot_states.critical_mpc_fail:
                    episode_reward = CRITICAL_FAIL_PENALTY  # Death Penalty
                    terminated = True
                else:
                    episode_reward += reward

                if terminated or truncated:
                    break

            # --- TRACK STAGE PROGRESS ---
            # Inspect the completion flags to see how far we got
            if hasattr(sim_instance.robot_sim, 'robot_states'):
                completed_dict = sim_instance.robot_sim.robot_states.sr_mode_completed
                # Find the highest key that is True
                for mode_idx, is_done in completed_dict.items():
                    if is_done:
                        current_run_max_stage = max(current_run_max_stage, mode_idx)

            stages_reached.append(current_run_max_stage)
            cumulative_raw_reward += episode_reward

        # --- AGGREGATION & COMPOSITE FITNESS ---

        # 1. Average Raw Reward (Quality of movement)
        avg_episode_reward = cumulative_raw_reward / len(initial_conditions)

        # 2. Average Stage Progress (Sequence discovery: -1.0 to 4.0)
        avg_task_progress = np.mean(stages_reached)

        # 3. Calculate Final Fitness for CEM
        # Shift progress so that -1 (Nothing) -> 0 Bonus, and 0 (Safe Mode) -> 1 Bonus
        progress_score = avg_task_progress + 1.0

        # Logic: If the robot is crashing (average <= -500), do NOT give it the progress bonus.
        # We don't want a robot that crashes at Stage 4 to beat a stable robot at Stage 0.
        if avg_episode_reward <= -500:
            final_fitness = avg_episode_reward
        else:
            final_fitness = avg_episode_reward + (progress_score * TASK_PROGRESS_WEIGHT)

        # 4. Get New Normalization Stats
        new_stats_object = sim_instance.normalizer.get_shadow_stats()

        # print(f"Process ID {sim_instance.env_id} Finished")

        # Return: (ID, Fitness, Stats, Metric_For_Curriculum)
        return task_id, final_fitness, new_stats_object, avg_task_progress

    except Exception as e:
        print(f"[Proc {os.getpid()}, Task {task_id}] ERROR in worker task: {e}")
        # print(traceback.format_exc()) # Uncomment for deep debugging

        # Return Worst-Case values on code failure
        return task_id, -float('inf'), None, -1.0
