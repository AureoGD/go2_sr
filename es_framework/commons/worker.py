import multiprocessing as mp
# Ensure start method is set (linux/mac default is fork, we usually want spawn for CUDA/Torch)
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

import os
import torch
import numpy as np
from typing import Dict, Any, Tuple

from es_framework.commons.nn_parameters import unflatten_nn_parameters
from es_framework.commons.control_rule import ControlRule
from environment.env_go2 import Go2Env

sim_instance = None
control_rule = None
is_discrete = True


def init_worker(config: Dict[str, Any]):
    global sim_instance, control_rule, is_discrete

    process_id = os.getpid()

    env_sw_config = config.get('env_sw_config', {})
    control_rule_cfg = config.get('model_config', {})

    is_discrete = control_rule_cfg.get('discrete', False)
    sim_instance = Go2Env(env_id=process_id, rendering=False)
    output_dim = sim_instance.action_space.n

    control_rule = ControlRule(observation_dim=sim_instance.observation_space.shape[0],
                               output_dim=output_dim,
                               **control_rule_cfg).reset_parameters(seed=45, last_layer_std=1)


def run_worker(task_args: Tuple[int, np.ndarray, list, float, Any]):
    global sim_instance, control_rule

    if sim_instance is None:
        raise RuntimeError("Worker not initialized correctly.")

    # Unpack the 5 arguments (norm_stats was missing in previous signature)
    task_id, nn_params_flat, initial_conditions, difficulty, norm_stats = task_args

    sim_instance.set_difficulty(difficulty)

    try:
        state_dict = unflatten_nn_parameters(nn_params_flat, control_rule)
        control_rule.load_state_dict(state_dict)

        # 1. SETUP: Load Global History into the Main Normalizer (The "Actor")
        # This ensures the robot sees the world correctly based on previous generations.
        if norm_stats is not None:
            mean_val, var_val, count_val = norm_stats
            # Access the main normalizer used for inference
            normalizer = sim_instance.normalizer.vel_normalizer

            normalizer.mean = mean_val
            normalizer.var = var_val
            normalizer.count = count_val

        # 2. SETUP: Reset the Shadow Normalizer (The "Collector")
        # This ensures we start counting from ZERO for this specific run.
        sim_instance.normalizer.reset_shadow()

        max_step = sim_instance.max_step
        cumulative_fitness = 0

        for initial_condition in initial_conditions:
            fitness = 0
            obs, info = sim_instance.reset(q0=initial_condition[0], r0=initial_condition[1], b0=initial_condition[2])

            for step in range(max_step):
                action, _ = control_rule.predict(obs)
                # Note: sim_instance.step calls normalizer.update(),
                # which now populates 'shadow_normalizer'
                obs, reward, terminated, truncated, info = sim_instance.step(action)
                fitness += reward
                if terminated:
                    break
                if truncated:
                    break
            cumulative_fitness += fitness

        # 3. RETURN: Send back ONLY the new stats (Shadow)
        # We do NOT return vel_normalizer stats, because that includes the history count.
        new_stats = (sim_instance.normalizer.shadow_normalizer.count, sim_instance.normalizer.shadow_normalizer.mean,
                     sim_instance.normalizer.shadow_normalizer.var)

        return task_id, cumulative_fitness / len(initial_conditions), new_stats

    except Exception as e:
        print(f"[Proc {os.getpid()}, Task {task_id}] ERROR in worker task: {e}")
        import traceback
        traceback.print_exc()
        # Return default small stats to prevent aggregation crash
        dummy_stats = (1e-4, np.zeros(18), np.ones(18))
        return task_id, -float('inf'), dummy_stats
