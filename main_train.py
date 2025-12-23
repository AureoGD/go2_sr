# =========================================================================
# 1. ENVIRONMENT SETTINGS (MUST BE AT THE VERY TOP)
# =========================================================================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm

# === PROJECT IMPORTS ===
from es_framework.optimizers import get_optimizer
from environment.learning_phases import LearningPhases
from es_framework.components.curriculum_manager import CurriculumManager
from es_framework.components.worker import init_worker, run_micro_task
from es_framework.components.policy import Policy
from es_framework.components.logger import TrainingLogger
from environment.go2_env import Go2Env


def main():
    # =========================================================================
    # 2. CONFIGURATION
    # =========================================================================
    config = {
        "optimizer_type": "CEM",
        "job_name": "go2_self_righting",
        "pop_size": 20,  # Number of unique parameter sets (candidates)
        "num_scenarios": 10,  # Number of shared postures per candidate
        "max_generations": 1000,
        "max_workers": 20,  # CPU core limit
        "sigma_init": 0.05,
        "sigma_decay": 0.995,
        "elite_frac": 0.2,
        "model_config": {
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "discrete": True
        }
    }

    print(f"{'='*60}")
    print(f"[Main] Initializing Training...")
    print(f"       Population: {config['pop_size']} | Scenarios: {config['num_scenarios']}")
    print(f"       Total Rollouts/Gen: {config['pop_size'] * config['num_scenarios']}")
    print(f"{'='*60}\n")

    # =========================================================================
    # 3. INITIALIZATION
    # =========================================================================
    # Probe dimensions using a dummy env
    dummy_env = Go2Env(rendering=False)
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.n if hasattr(dummy_env.action_space, 'n') else dummy_env.action_space.shape[0]
    dummy_env.close()

    policy = Policy(obs_dim, act_dim, **config["model_config"])
    phases = LearningPhases()
    curriculum = CurriculumManager(phases=phases)
    optimizer = get_optimizer(config['optimizer_type'], config, policy)

    # Initialize Unified Logger (Saves Config, TB, CSV, and Models)
    logger = TrainingLogger(config=config, root_dir="results", periodic_interval=10)
    logger.set_reference_model(policy)

    global_stats = None
    num_workers = min(config['max_workers'], config['pop_size'])

    # =========================================================================
    # 4. TRAINING LOOP
    # =========================================================================
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(config,)) as pool:

        for gen in range(config['max_generations']):
            gen_start_time = time.time()

            # --- STEP 1: ASK (Generate Population) ---
            candidates = optimizer.ask()

            # --- STEP 2: PREPARE EVALUATION ---
            # Generate shared scenarios for fair comparison (10 scenarios)
            shared_scenarios = curriculum.get_reset_conditions(n_scenarios=config['num_scenarios'])
            difficulty_level = curriculum.get_difficulty()

            tasks = []
            for i in range(config['pop_size']):
                for s_idx in range(config['num_scenarios']):
                    # Task: (candidate_id, params, scenario, difficulty, stats)
                    task = (i, candidates[i], shared_scenarios[s_idx], difficulty_level, global_stats)
                    tasks.append(task)

            # --- STEP 3: EXECUTE (Parallel Rollouts) ---
            all_results = []
            with tqdm(total=len(tasks), desc=f"Gen {gen:3d}", leave=False) as pbar:
                for res in pool.imap(run_micro_task, tasks):
                    all_results.append(res)

                    # Update live average on the progress bar every 10 tasks
                    if len(all_results) % 10 == 0:
                        current_avg = np.mean([r[1] for r in all_results])
                        pbar.set_postfix({"live_avg": f"{current_avg:.1f}"})
                    pbar.update(1)

            # --- STEP 4: AGGREGATE RESULTS ---
            candidate_rewards = {i: [] for i in range(config['pop_size'])}
            candidate_stages = {i: [] for i in range(config['pop_size'])}
            worker_stats_list = []

            for res in all_results:
                c_id, reward, stats, max_stage = res
                candidate_rewards[c_id].append(reward)
                candidate_stages[c_id].append(max_stage)
                if stats is not None:
                    worker_stats_list.append(stats)

            # Average fitness per candidate across all scenarios
            rewards_np = np.array([np.mean(candidate_rewards[i]) for i in range(config['pop_size'])])

            # Calculate Success Rate based on candidates consistently reaching the goal (Stage 3+)
            avg_stages = [np.mean(candidate_stages[i]) for i in range(config['pop_size'])]
            success_rate = np.mean([1.0 if s >= 3.0 else 0.0 for s in avg_stages])

            # --- STEP 5: TELL (Optimizer Update) ---
            optimizer.tell(candidates, rewards_np)

            # --- STEP 6: CURRICULUM & STATS UPDATE ---
            if worker_stats_list:
                # Update global normalizer stats (using first worker's shadow stats for stability)
                global_stats = worker_stats_list[0]

            curriculum_updated = curriculum.update(success_rate, optimizer)

            # --- STEP 7: LOGGING ---
            gen_duration = time.time() - gen_start_time

            # ETA Calculation
            remaining_gens = config['max_generations'] - gen - 1
            eta_hrs = (gen_duration * remaining_gens) / 3600

            # 1. Prepare population for logger [(params, reward), ...]
            evaluated_pop = [(candidates[i], rewards_np[i]) for i in range(config['pop_size'])]

            # 2. Package all extra info for TensorBoard, CSV, and Console
            extra_metrics = {
                "success_rate": success_rate,
                "difficulty": difficulty_level,
                "sigma": optimizer.sigma,
                "eta": eta_hrs,
                "gen_duration": gen_duration
            }

            # 3. Log Generation (Handles console print, CSV, TB, and .pth saving)
            logger.log_generation(generation=gen,
                                  evaluated_population=evaluated_pop,
                                  optimizer=optimizer,
                                  normalization_stats=global_stats,
                                  extra_metrics=extra_metrics)

    # Final cleanup
    logger.close()


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    main()
