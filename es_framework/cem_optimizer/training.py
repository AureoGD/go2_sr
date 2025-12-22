import os
import torch
import multiprocessing
import numpy as np
from functools import partial
from typing import Tuple, List, Optional
from tqdm import tqdm

# Set thread limits to avoid CPU contention (Redundant safety, as worker does it too)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# --- Custom Imports ---
from es_framework.commons.worker import init_worker, run_worker
from es_framework.commons.nn_parameters import flatten_nn_parameters
from es_framework.commons.logger import TrainingLogger
from es_framework.cem_optimizer.cem_optimizer import CEMOptimizer
from es_framework.commons.control_rule import ControlRule

# --- Environment & Curriculum Imports ---
from environment.learning_phases import LearningPhases
from environment.go2_state_normalizer import RunningNormalizer, NormalizerStats
from es_framework.commons.curriculum_manager import CurriculumManager

# --- Hyperparameters ---
is_discrete = True
POPULATION_SIZE = 20
GENERATIONS = 150
ELITE_FRACTION = 0.25
INITIAL_STD_DEV = 2.5
EXTRA_NOISE_SCALE = 0.5
NOISE_DECAY_FACTOR = 0.99
MIN_STD_DEV = 0.001
UPDATE_RULE = "standard"
ELITE_WEIGHTING = "uniform"

# Neural Net Params
fc1_dim = 128
fc2_dim = 128

# Logger Params
log_dir = "es_framework"
algorithm = 'cem'


def main():
    # 1. Initialize Logger
    logger = TrainingLogger(discrete=is_discrete, alg=algorithm)

    # 2. Model Configuration
    config = {'model_config': {'fc1_dim': fc1_dim, 'fc2_dim': fc2_dim, 'discrete': is_discrete}}
    config['is_discrete'] = True

    reference_model = ControlRule(observation_dim=63, output_dim=5, **config['model_config'])
    logger.set_reference_model(reference_model)
    param_dim = flatten_nn_parameters(reference_model).size

    # 3. Optimizer Setup
    cem = CEMOptimizer(param_dim=param_dim,
                       population_size=POPULATION_SIZE,
                       elite_fraction=ELITE_FRACTION,
                       initial_std_dev=INITIAL_STD_DEV,
                       update_rule_type=UPDATE_RULE,
                       elite_weighting_type=ELITE_WEIGHTING,
                       noise_decay_factor=NOISE_DECAY_FACTOR,
                       min_std_dev=MIN_STD_DEV,
                       extra_noise_scale=EXTRA_NOISE_SCALE)
    cem.set_initial_mean_params(reference_model)

    # 4. Parallel Workers Setup
    num_workers = min(20, POPULATION_SIZE)
    logger.log_generation(0, [], None, None)  # Init console
    print(f"[System] Starting CEM training with {num_workers} parallel workers.")

    initializer_with_args = partial(init_worker, config=config)
    pool = multiprocessing.Pool(processes=num_workers, initializer=initializer_with_args)

    # 5. Curriculum & Data Managers
    learning_phases = LearningPhases()
    curriculum_mgr = CurriculumManager(phases=learning_phases, plateau_patience=10, consistency_threshold=0.8)

    try:
        # Holds the global normalization statistics (NormalizerStats object)
        current_global_stats: Optional[NormalizerStats] = None

        for gen in range(1, GENERATIONS + 1):

            # A. Sample Population
            population_params = cem.sample_population()

            # B. Curriculum: Determine Stage & Generate Data
            active_stage_idx = curriculum_mgr.current_max_stage

            # Generates a MIXED batch (Frontier + History)
            initial_conditions, _ = learning_phases.get_initial_conditions(num_conditions=10,
                                                                           max_difficulty_id=active_stage_idx)

            active_config = curriculum_mgr.frontier_config
            diff_id = active_config.difficulty_id
            current_slope_rad = active_config.terrain_angle_rad[1]

            # C. Dispatch Tasks
            tasks = [(i, params, initial_conditions, diff_id, current_global_stats)
                     for i, params in enumerate(population_params)]

            # --- PROGRESS BAR LOGIC ---
            results = []
            # 'imap_unordered' yields results as soon as they finish
            with tqdm(total=len(tasks), desc=f"Gen {gen:03d} Processing", unit="ind", leave=False) as pbar:
                for res in pool.imap_unordered(run_worker, tasks):
                    results.append(res)
                    pbar.update(1)

            # CRITICAL: Sort results by task_id (index 0) because imap_unordered scrambles them
            results.sort(key=lambda x: x[0])

            # D. Process Results
            fitness_scores = [r[1] for r in results]

            # Extract 'Shadow Stats' (new data collected by workers)
            workers_stats_list = [r[2] for r in results]

            # Extract NEW Metric: Average Task Progress (-1.0 to 4.0)
            task_progress_scores = [r[3] for r in results]

            # E. Aggregation Logic
            batch_stats = RunningNormalizer.aggregate(workers_stats_list)

            if batch_stats is None:
                print(f"\n[Warning] Gen {gen}: All workers returned invalid stats.")
                # Fallback: Init empty stats if this is the first gen and it failed
                if current_global_stats is None:
                    current_global_stats = NormalizerStats(count=1e-4,
                                                           mean=np.zeros(63, dtype=np.float32),
                                                           var=np.ones(63, dtype=np.float32))
            else:
                # Merge Batch with Global History
                if current_global_stats is None:
                    current_global_stats = batch_stats
                else:
                    current_global_stats = RunningNormalizer.aggregate([current_global_stats, batch_stats])

            # F. Update Optimizer
            evaluated_population = list(zip(population_params, fitness_scores))
            cem.update_distribution(evaluated_population)

            # G. Check Curriculum Progression
            # Pass the task_progress_scores to the manager
            scenario_ids_proxy = [active_stage_idx] * len(evaluated_population)

            curriculum_mgr.check_progression(
                population_data=evaluated_population,
                scenario_ids=scenario_ids_proxy,
                population_task_progress=task_progress_scores,  # <--- NEW ARGUMENT
                optimizer=cem)

            # H. Log Generation
            # Calculate metrics for logging
            best_idx = np.argmax(fitness_scores)
            best_ind_depth = task_progress_scores[best_idx]
            pop_mean_depth = np.mean(task_progress_scores)

            logger.log_generation(
                generation=gen,
                evaluated_population=evaluated_population,
                normalization_stats=current_global_stats,
                extra_metrics={
                    "cem_sigma_mean": float(getattr(cem, "mean_std_devs", 0.0)),
                    "cem_epsilon": getattr(cem, "epsilon", 0.0),
                    "difficulty_id": diff_id,
                    "frontier_slope": current_slope_rad,
                    # New Data for Tensorboard:
                    "Best_Ind_Avg_Depth": best_ind_depth,
                    "Pop_Mean_Avg_Depth": pop_mean_depth
                })

    except KeyboardInterrupt:
        print("\n[System] Training interrupted by user.")

    except Exception as e:
        print(f"\n[Error] Training crashed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        pool.close()
        pool.join()

        # Clean Save
        if current_global_stats is not None:
            logger.save_checkpoint(filename="model_final.pth",
                                   params=cem.get_best_params(),
                                   norm_stats=current_global_stats,
                                   meta={
                                       'fitness': 'final',
                                       'gen': GENERATIONS
                                   })

        logger.close()


if __name__ == "__main__":
    main()
