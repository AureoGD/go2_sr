import os

# Set thread limits before importing torch/numpy to avoid CPU contention
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import multiprocessing
from functools import partial
import numpy as np
from typing import Tuple, List, Optional

# --- Custom Imports ---
# Ensure these paths match your project structure
from es_framework.commons.worker import init_worker, run_worker
from es_framework.commons.nn_parameters import flatten_nn_parameters, unflatten_nn_parameters
from es_framework.commons.logger import TrainingLogger
from es_framework.cem_optimizer.cem_optimizer import CEMOptimizer
from es_framework.commons.control_rule import ControlRule
from es_framework.commons.initial_conditions import SelfAdaptingCurriculum

# --- Experiment Setup ---
is_discrete = True
# --- CEM Hyperparameters ---
POPULATION_SIZE = 20
GENERATIONS = 150
ELITE_FRACTION = 0.25
INITIAL_STD_DEV = 2.5
EXTRA_NOISE_SCALE = 0.5
NOISE_DECAY_FACTOR = 0.99
MIN_STD_DEV = 0.001
UPDATE_RULE = "standard"  # "standard" or "cmaes_type"
ELITE_WEIGHTING = "uniform"  # "uniform" or "logarithmic"

# --- Neural Net Setup ---
fc1_dim = 128
fc2_dim = 128

# --- Logging Setup ---
log_dir = "es_framework"
algorithm = 'cem'


def aggregate_running_stats(stats_list: List[Tuple[float, np.ndarray, np.ndarray]]):
    """
    Merges a list of (count, mean, var) tuples using Chan's parallel algorithm.
    Ignores any workers that return NaN or Inf values.
    
    Args:
        stats_list: List of tuples (count, mean, var)
        
    Returns:
        (total_count, final_mean, final_var) or (None, None, None) if all failed.
    """
    if not stats_list:
        return 0, 0.0, 1.0  # Return default safe values if list is empty

    # Initialize with clean defaults
    total_n = 0
    grand_mean = None
    grand_m2 = None

    valid_workers_count = 0

    for i, (n_b, mu_b, var_b) in enumerate(stats_list):
        # --- SAFETY CHECK ---
        # If any value is NaN or Infinite, skip this worker
        if not (np.isfinite(n_b) and np.all(np.isfinite(mu_b)) and np.all(np.isfinite(var_b))):
            # Optional: Print warning so you know a worker failed
            # print(f"Warning: Worker {i} returned invalid stats (NaN/Inf). Skipping.")
            continue

        # Calculate M2 (Sum of Squares) for the current worker
        m2_b = var_b * n_b

        # If this is the first VALID worker found, initialize
        if grand_mean is None:
            total_n = n_b
            grand_mean = mu_b
            grand_m2 = m2_b
            valid_workers_count += 1
            continue

        # Standard Chan's Algorithm Update
        n_new = total_n + n_b
        delta = mu_b - grand_mean

        new_mean = grand_mean + delta * (n_b / n_new)
        new_m2 = grand_m2 + m2_b + (delta**2) * (total_n * n_b / n_new)

        total_n = n_new
        grand_mean = new_mean
        grand_m2 = new_m2
        valid_workers_count += 1

    # If NO workers were valid (catastrophic failure of all), return None
    if grand_mean is None:
        return None, None, None

    final_var = grand_m2 / total_n
    return total_n, grand_mean, final_var


def main():
    # Initialize Logger
    logger = TrainingLogger(discrete=is_discrete, alg=algorithm)

    # --- Environment / Model Config ---
    config = {'model_config': {'fc1_dim': fc1_dim, 'fc2_dim': fc2_dim, 'discrete': is_discrete}}
    config['is_discrete'] = True

    # Setup Reference Model for dimensions
    reference_model = ControlRule(observation_dim=63, output_dim=5, **config['model_config'])
    logger.set_reference_model(reference_model)
    param_dim = flatten_nn_parameters(reference_model).size

    # Initialize Optimizer
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

    # --- Parallel Setup ---
    num_workers = min(20, POPULATION_SIZE)
    logger.log_message(f"Starting CEM training with {num_workers} persistent parallel workers.")

    # Initialize workers with config
    initializer_with_args = partial(init_worker, config=config)
    pool = multiprocessing.Pool(processes=num_workers, initializer=initializer_with_args)

    # Curriculum Setup
    curriculum = SelfAdaptingCurriculum(min_difficulty=0.1, max_difficulty=1.0, use_ema_variance=True)

    try:
        # Variable to hold the Global History of normalization statistics
        # Structure: (mean, var, count) -> Matched to run_worker unpacking
        current_norm_stats = None

        for gen in range(1, GENERATIONS + 1):

            # 1. Sample Population
            population_params = cem.sample_population()

            # 2. Update Curriculum
            curriculum.set_difficulty(1.0)
            initial_conditions = curriculum.get_initial_conditions(10)
            diff = curriculum.current_difficulty
            slope = curriculum.slope

            # 3. Dispatch Tasks
            # We pass 'current_norm_stats' (History) so workers can normalize inputs correctly
            tasks = [
                (i, params, initial_conditions, diff, current_norm_stats) for i, params in enumerate(population_params)
            ]

            results = pool.map(run_worker, tasks)
            results.sort(key=lambda x: x[0])

            # 4. Process Results
            fitness_scores = [r[1] for r in results]

            # These are 'Shadow' stats from the workers: (count, mean, var)
            # They represent ONLY the data collected in THIS generation.
            workers_new_stats = [r[2] for r in results]

            # --- AGGREGATION LOGIC ---

            # Step A: Aggregate the Batch (Combine all workers for this Gen)
            # Returns: (count, mean, var)
            batch_count, batch_mean, batch_var = aggregate_running_stats(workers_new_stats)

            if batch_mean is None:
                print(f"Gen {gen}: WARN - All workers returned invalid stats. Skipping normalization update.")

                # Fallback: If we have no history yet and batch failed, create a dummy init
                if current_norm_stats is None:
                    # (mean, var, count)
                    current_norm_stats = (np.zeros(18), np.ones(18), 1e-4)
            else:
                # Step B: Merge Batch with History
                if current_norm_stats is None:
                    # First generation: History IS the batch
                    # Save as (Mean, Var, Count) to match run_worker expectation
                    current_norm_stats = (batch_mean, batch_var, batch_count)
                else:
                    # Unpack History (stored as Mean, Var, Count)
                    hist_mean, hist_var, hist_count = current_norm_stats

                    # Merge History + Batch using the same Chan's algorithm
                    # input list expects: (Count, Mean, Var)
                    new_total_count, new_global_mean, new_global_var = aggregate_running_stats([
                        (hist_count, hist_mean, hist_var),  # History
                        (batch_count, batch_mean, batch_var)  # New Batch
                    ])

                    # Update History: Store as (Mean, Var, Count)
                    current_norm_stats = (new_global_mean, new_global_var, new_total_count)

            # Unpack for logging
            global_mean, global_var, total_count = current_norm_stats

            # --- END AGGREGATION ---

            # 5. Update Distribution (CEM Logic)
            evaluated_population = list(zip(population_params, fitness_scores))
            cem.update_distribution(evaluated_population)
            curriculum.update_difficulty(evaluated_population)

            # 6. Log Generation
            # Pass the Global History stats to be saved to disk
            logger.log_generation(generation=gen,
                                  evaluated_population=evaluated_population,
                                  extra_metrics={
                                      "Mean_StdDev_Params": float(getattr(cem, "mean_std_devs", float('nan'))),
                                      "Extra_Noise_Scale": getattr(cem, "epsilon", float('nan')),
                                      "Difficulty": diff,
                                      "Slope": slope
                                  },
                                  normalization_data={
                                      "mean": global_mean,
                                      "var": global_var,
                                      "count": total_count
                                  })

    except KeyboardInterrupt:
        logger.log_message("Training interrupted by user.")

    except Exception as e:
        logger.log_message(f"Training crashed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        logger.log_message("Closing worker pool and saving final model...")
        pool.close()
        pool.join()

        # Save Final Model
        final_weights = cem.get_best_params()
        final_state_dict = unflatten_nn_parameters(final_weights, reference_model)
        final_path = os.path.join(logger.models_save_dir, "cem_model_final_mean.pth")

        torch.save(final_state_dict, final_path)

        # Also save the final normalizer stats associated with this model
        if current_norm_stats is not None:
            final_norm_path = os.path.join(logger.models_save_dir, "cem_model_final_mean_normalizer.npz")
            np.savez(final_norm_path,
                     mean=current_norm_stats[0],
                     var=current_norm_stats[1],
                     count=current_norm_stats[2])
            logger.log_message(f"Final normalizer stats saved to {final_norm_path}")

        logger.log_message(f"Final CEM mean weights saved to {final_path}")
        logger.close()


if __name__ == "__main__":
    main()
