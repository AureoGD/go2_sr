import numpy as np
from collections import deque
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from es_framework.cem_optimizer.cem_optimizer import CEMOptimizer
    from environment.learning_phases import LearningPhases


class CurriculumManager:

    def __init__(self,
                 phases: 'LearningPhases',
                 plateau_patience: int = 10,
                 consistency_threshold: float = 0.8,
                 target_task_idx: float = 4.0):  # Target is completion of Stage 4 (Stand Up)

        self.phases = phases
        self.plateau_window_size = plateau_patience
        self.consistency_threshold = consistency_threshold
        self.target_task_idx = target_task_idx

        # Tolerance: 3.6 means "Elites reach stage 4 most of the time"
        # (Average 3.6 implies 90% success rate, or very consistent deep progress)
        self.task_tolerance = 0.4
        self.plateau_tolerance = 0.05

        self.current_max_stage = 0
        self.score_history = deque(maxlen=self.plateau_window_size)
        self.consecutive_ready_signals = 0

    @property
    def frontier_config(self):
        return self.phases.scenarios[self.current_max_stage]

    def check_progression(
            self,
            population_data: list,
            scenario_ids: list,
            population_task_progress: list,  # List of floats (e.g. -1.0 to 4.0)
            optimizer: 'CEMOptimizer') -> bool:

        # 1. Filter Data for Current Frontier Stage
        all_scores = np.array([fit for _, fit in population_data])
        all_progress = np.array(population_task_progress)
        scenario_ids = np.array(scenario_ids)

        mask = (scenario_ids == self.current_max_stage)
        frontier_scores = all_scores[mask]
        frontier_progress = all_progress[mask]

        if len(frontier_scores) < 4:
            return False

        # 2. Sort by Fitness (Find Elites)
        # We rely on fitness to identify the "smart" robots, then check their progress.
        n_elites = max(1, int(len(frontier_scores) * 0.25))
        sort_idx = np.argsort(frontier_scores)[::-1]

        # Extract Elite Data
        elite_scores = frontier_scores[sort_idx][:n_elites]
        elite_progress = frontier_progress[sort_idx][:n_elites]

        rest_scores = frontier_scores[sort_idx][n_elites:]
        if len(rest_scores) == 0:
            rest_scores = elite_scores

        mean_elite = np.mean(elite_scores)
        mean_rest = np.mean(rest_scores)

        # 3. Calculate Metrics

        # Metric A: Elite Depth (Sequence Discovery)
        # "On average, how far down the task list do the elites get?"
        avg_elite_depth = np.mean(elite_progress)

        # Metric B: Fitness Consistency
        if abs(mean_elite) < 1e-3:
            consistency_ratio = 1.0
        else:
            consistency_ratio = mean_rest / mean_elite

        # Metric C: Plateau Detection
        self.score_history.append(mean_elite)
        is_plateaued = False
        rate_of_change = 0.0

        if len(self.score_history) == self.plateau_window_size:
            start_score = self.score_history[0]
            current_score = self.score_history[-1]
            if abs(start_score) > 1e-3:
                rate_of_change = abs(current_score - start_score) / abs(start_score)

            # If fitness stops changing by more than 5%, we have plateaued
            if rate_of_change < self.plateau_tolerance:
                is_plateaued = True

        # --- DECISION LOGIC ---

        # 1. Task Completion Check
        # We only advance if elites are consistently reaching near the target (Stage 4)
        target_threshold = self.target_task_idx - self.task_tolerance
        is_task_done = avg_elite_depth >= target_threshold

        # 2. Consistency Check
        is_consistent = consistency_ratio >= self.consistency_threshold

        print(f"   >>> [Curriculum] Stage {self.current_max_stage} Check:")
        print(
            f"       Elite Avg Depth: {avg_elite_depth:.2f} / {self.target_task_idx} -> {'PASS' if is_task_done else 'FAIL'}"
        )
        print(f"       Fitness Stability: {rate_of_change*100:.1f}% -> {'PLATEAU' if is_plateaued else 'CHANGING'}")

        # REQUIREMENT: Consistent AND Plateaued AND Task Completed
        if is_consistent and is_plateaued and is_task_done:
            self.consecutive_ready_signals += 1
        else:
            self.consecutive_ready_signals = 0

        if self.consecutive_ready_signals >= 3:
            return self._advance_stage(optimizer)

        return False

    def _advance_stage(self, optimizer):
        if self.current_max_stage >= len(self.phases.scenarios) - 1:
            print("   >>> [Curriculum] MAX DIFFICULTY REACHED.")
            return False

        self.current_max_stage += 1
        self.score_history.clear()
        self.consecutive_ready_signals = 0

        # Expand exploration slightly when entering new terrain
        if hasattr(optimizer, 'sigma'):
            optimizer.sigma = np.maximum(optimizer.sigma, 1.5)

        print(f"\n{'='*60}")
        print(f"LEVEL UP! Unlocked Stage {self.current_max_stage}: {self.frontier_config.name}")
        print(f"{'='*60}\n")
        return True
