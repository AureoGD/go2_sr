import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
from tqdm import tqdm

# Conditional import for TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from es_framework.components.nn_utils import unflatten_nn_parameters


class TrainingLogger:

    def __init__(self, config: Dict[str, Any], root_dir: str = "results", periodic_interval: int = 10):
        """
        Unified Logger for Go2 Self-Righting.
        Creates a single folder per run containing models, logs, and config.
        """
        self.alg = config.get('optimizer_type', 'ES')
        self.job_name = config.get('job_name', 'go2_sr')
        self.periodic_interval = periodic_interval
        self.start_time = time.time()
        self.last_gen_time = self.start_time

        # --- 1. Create Unified Run Directory ---
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.job_name}_{self.alg}_{time_str}"
        self.base_path = Path(root_dir) / self.run_id

        self.model_dir = self.base_path / "models"
        self.tb_dir = self.base_path / "tensorboard"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        # --- 2. Save Config as JSON ---
        self._save_config(config)

        # --- 3. TensorBoard Setup ---
        self.tb_writer = None
        if SummaryWriter:
            self.tb_writer = SummaryWriter(log_dir=str(self.tb_dir))

        # --- 4. CSV Setup ---
        self.csv_path = self.base_path / "history.csv"
        self.csv_file = None
        self.csv_writer = None
        self.csv_fieldnames = [
            "generation", "timestamp", "duration", "fitness_best", "fitness_mean", "success_rate", "difficulty",
            "sigma", "max_stage_reached", "eta"
        ]

        # --- 5. State ---
        self.overall_best_fitness = -np.inf
        self.reference_model = None

    def set_reference_model(self, model: torch.nn.Module):
        self.reference_model = model

    def log_generation(self,
                       generation: int,
                       evaluated_population: List[Tuple[np.ndarray, float]],
                       optimizer: Any,
                       normalization_stats: Optional[Any] = None,
                       extra_metrics: Optional[Dict[str, float]] = None):
        if not evaluated_population:
            return

        current_time = time.time()
        duration = current_time - self.last_gen_time
        self.last_gen_time = current_time

        fitnesses = [score for _, score in evaluated_population]
        best_gen_fitness = np.max(fitnesses)
        best_gen_params = evaluated_population[np.argmax(fitnesses)][0]

        stats = {
            "generation": generation,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "duration": round(duration, 2),
            "fitness_best": round(best_gen_fitness, 4),
            "fitness_mean": round(np.mean(fitnesses), 4),
            "overall_best": round(max(self.overall_best_fitness, best_gen_fitness), 4)
        }

        if extra_metrics:
            stats.update(extra_metrics)

        # Check for new overall best
        if best_gen_fitness > self.overall_best_fitness:
            self.overall_best_fitness = best_gen_fitness
            self.save_checkpoint("best_overall.pth", best_gen_params, normalization_stats, optimizer, generation)

        # Periodic checkpoint
        if generation % self.periodic_interval == 0:
            self.save_checkpoint(f"gen_{generation:04d}.pth", best_gen_params, normalization_stats, optimizer,
                                 generation)

        # --- PERSIST TO CSV ---
        self._write_csv(stats)

        # --- TENSORBOARD LOGGING (With Grouping Fixes) ---
        if self.tb_writer:
            for k, v in stats.items():
                try:
                    # Force conversion to float for NumPy/Tensor/Scalar compatibility
                    val = float(v)

                    # Logic-based grouping for cleaner UI
                    if "fitness" in k:
                        tag = f"Fitness/{k.replace('fitness_', '')}"
                    elif "sigma" in k:
                        tag = f"Optimizer/Sigma"
                    elif "stage_dist" in k:
                        # Puts stage_dist/0_go_safe etc. into a "Stages" folder
                        tag = f"Stages/{k.split('/')[-1]}"
                    elif k in ["success_rate", "difficulty", "max_stage_reached"]:
                        tag = f"Curriculum/{k}"
                    else:
                        tag = f"Metrics/{k}"

                    self.tb_writer.add_scalar(tag, val, generation)
                except (TypeError, ValueError):
                    # Skips timestamp or other non-numeric strings
                    continue

    def save_checkpoint(self, filename: str, params: np.ndarray, norm_stats: Any, optimizer: Any, gen: int):
        if self.reference_model is None:
            return

        save_path = self.model_dir / filename
        state_dict = unflatten_nn_parameters(params, self.reference_model)

        if hasattr(norm_stats, 'mean'):
            norm_data = {'mean': norm_stats.mean, 'var': norm_stats.var}
        else:
            norm_data = norm_stats if norm_stats else {}
            norm_data = None

        checkpoint = {
            'generation': gen,
            'model_state_dict': state_dict,
            'optimizer_state': {
                'mean': optimizer.mean,
                'sigma': optimizer.sigma
            },
            'normalizer_state': norm_data
        }
        torch.save(checkpoint, save_path)

    def _save_config(self, config: Dict):
        with open(self.base_path / "config.json", 'w') as f:
            json.dump(config, f, indent=4)

    def _write_csv(self, stats: Dict):
        if self.csv_file is None:
            # Dynamically add any extra metrics to CSV header
            for k in stats.keys():
                if k not in self.csv_fieldnames:
                    self.csv_fieldnames.append(k)
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
            self.csv_writer.writeheader()

        row = {k: v for k, v in stats.items() if k in self.csv_fieldnames}
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        if self.csv_file:
            self.csv_file.close()
        if self.tb_writer:
            self.tb_writer.close()
        tqdm.write(f"\n[Logger] Results saved at: {self.base_path}")
