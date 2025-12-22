import os
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
import torch

# Conditional import to prevent crashes if TensorBoard is missing
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from es_framework.commons.nn_parameters import unflatten_nn_parameters


class TrainingLogger:

    def __init__(self,
                 alg: str = 'cmaes',
                 discrete: bool = False,
                 log_to_csv: bool = True,
                 log_to_tensorboard: bool = True,
                 save_overall_best_model: bool = True,
                 save_periodic_best_model: bool = True,
                 periodic_best_model_interval: int = 50,
                 root_log_dir: str = "logs",
                 root_model_dir: str = "models"):
        self.alg = alg
        self.start_time = time.time()
        self.last_gen_time = self.start_time

        # --- ID & Path Setup ---
        prefix = 'D_' if discrete else 'C_'
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{prefix}{self.alg}_{time_str}"

        self.log_dir = Path(root_log_dir) / alg / run_id
        self.model_dir = Path(root_model_dir) / alg / run_id

        self.log_dir.mkdir(parents=True, exist_ok=True)
        if save_overall_best_model or save_periodic_best_model:
            self.model_dir.mkdir(parents=True, exist_ok=True)

        # --- TensorBoard ---
        self.tb_writer = None
        if log_to_tensorboard and SummaryWriter:
            print(f"[TensorBoard] Logging to: {self.log_dir}")
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))

        # --- CSV Setup ---
        self.log_to_csv = log_to_csv
        self.csv_file = None
        self.csv_writer = None
        self.csv_fieldnames = [
            "generation", "timestamp", "duration", "pop_size", "fitness_best", "fitness_mean", "fitness_std",
            "fitness_min", "overall_best"
        ]

        # --- State ---
        self.save_overall = save_overall_best_model
        self.save_periodic = save_periodic_best_model
        self.periodic_interval = periodic_best_model_interval
        self.overall_best_fitness = -np.inf
        self.reference_model = None

    def set_reference_model(self, model: torch.nn.Module):
        """Required for unflattening parameters before saving."""
        self.reference_model = model

    def log_generation(self,
                       generation: int,
                       evaluated_population: List[Tuple[np.ndarray, float]],
                       normalization_stats: Optional[Any] = None,
                       extra_metrics: Optional[Dict[str, float]] = None):

        # --- FIX: handle empty population (Dummy Call) ---
        if not evaluated_population:
            print(f"[System] Logger initialized. Ready for Gen {generation}.")
            return

        current_time = time.time()
        duration = current_time - self.last_gen_time
        self.last_gen_time = current_time
        timestamp_str = datetime.now().strftime("%H:%M:%S")

        # 1. Calculate Statistics
        fitnesses = [score for _, score in evaluated_population]

        # Safety check for all-NaN fitness to prevent crash
        if len(fitnesses) == 0:
            print(f"[Warning] Gen {generation}: Population is empty during calc.")
            return

        best_gen_fitness = np.max(fitnesses)
        best_gen_params = evaluated_population[np.argmax(fitnesses)][0]

        stats = {
            "generation": generation,
            "timestamp": timestamp_str,
            "duration": round(duration, 2),
            "pop_size": len(fitnesses),
            "fitness_best": round(best_gen_fitness, 4),
            "fitness_mean": round(np.mean(fitnesses), 4),
            "fitness_std": round(np.std(fitnesses), 4),
            "fitness_min": round(np.min(fitnesses), 4),
            "overall_best": round(max(self.overall_best_fitness, best_gen_fitness), 4)
        }

        # 2. Merge Extra Metrics
        if extra_metrics:
            stats.update(extra_metrics)

        # 3. Handle Saving Models
        if best_gen_fitness > self.overall_best_fitness:
            self.overall_best_fitness = best_gen_fitness
            if self.save_overall:
                self.save_checkpoint(filename="model_best_overall.pth",
                                     params=best_gen_params,
                                     norm_stats=normalization_stats,
                                     meta={
                                         'fitness': best_gen_fitness,
                                         'gen': generation
                                     })

        if self.save_periodic and (generation % self.periodic_interval == 0):
            self.save_checkpoint(filename=f"model_gen_{generation:04d}.pth",
                                 params=best_gen_params,
                                 norm_stats=normalization_stats,
                                 meta={
                                     'fitness': best_gen_fitness,
                                     'gen': generation
                                 })

        # 4. Write to CSV
        if self.log_to_csv:
            self._write_csv(stats)

        # 5. Write to TensorBoard
        if self.tb_writer:
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    if k.startswith("fitness"):
                        self.tb_writer.add_scalar(f"Fitness/{k.replace('fitness_', '')}", v, generation)
                    else:
                        self.tb_writer.add_scalar(f"Metrics/{k}", v, generation)

        # 6. Console Print
        print(f"Gen {generation:03d} | Best: {stats['fitness_best']} | Mean: {stats['fitness_mean']} | "
              f"Std: {stats['fitness_std']} | {timestamp_str}")

    def save_checkpoint(self, filename: str, params: np.ndarray, norm_stats: Any, meta: Dict = None):
        if self.reference_model is None:
            print("[Warning] Reference model not set. Cannot save checkpoint.")
            return

        save_path = self.model_dir / filename
        state_dict = unflatten_nn_parameters(params, self.reference_model)

        norm_data = {}
        if norm_stats is not None:
            if hasattr(norm_stats, 'mean'):
                norm_data = {'mean': norm_stats.mean, 'var': norm_stats.var, 'count': norm_stats.count}
            elif isinstance(norm_stats, dict):
                norm_data = norm_stats

        checkpoint = {'model_state_dict': state_dict, 'normalizer_state': norm_data, 'metadata': meta or {}}

        try:
            torch.save(checkpoint, save_path)
        except Exception as e:
            print(f"[Error] Failed to save checkpoint: {e}")

    def _write_csv(self, stats: Dict):
        if self.csv_file is None:
            current_keys = list(stats.keys())
            for k in current_keys:
                if k not in self.csv_fieldnames:
                    self.csv_fieldnames.append(k)

            try:
                self.csv_file = open(self.log_dir / "history.csv", 'w', newline='')
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
                self.csv_writer.writeheader()
            except IOError as e:
                print(f"[Error] CSV Init failed: {e}")
                self.log_to_csv = False
                return

        if self.csv_writer:
            clean_stats = {k: v for k, v in stats.items() if k in self.csv_fieldnames}
            self.csv_writer.writerow(clean_stats)
            self.csv_file.flush()

    def close(self):
        if self.csv_file:
            self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()

        print(f"\n[System] Training Complete.")
        print(f"   Logs:   {self.log_dir}")
        print(f"   Models: {self.model_dir}")
