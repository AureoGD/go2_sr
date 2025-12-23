import numpy as np
import torch
from typing import List, Tuple, Union
from es_framework.components.nn_utils import flatten_nn_parameters


class CEMOptimizer:

    def __init__(self, config: dict, policy_structure: torch.nn.Module):
        """
        Cross-Entropy Method Optimizer with advanced update rules.
        
        Args:
            config (dict): Configuration dictionary containing hyperparameters.
            policy_structure (torch.nn.Module): The neural network model (used to get param dims).
        """
        self.policy = policy_structure

        # 1. Determine Parameter Dimension
        # We try to use the utility, but fallback to counting manually if needed
        try:
            self.mean = flatten_nn_parameters(self.policy)
            self.param_dim = self.mean.shape[0]
        except:
            self.param_dim = sum(p.numel() for p in self.policy.parameters())
            self.mean = np.zeros(self.param_dim, dtype=np.float32)

        # 2. Extract Config
        self.population_size = config.get('pop_size', 50)
        elite_fraction = config.get('elite_frac', 0.2)
        initial_std_dev = config.get('sigma_init', 0.1)

        # Advanced Config (with defaults)
        self.update_rule_type = config.get('update_rule', "standard")  # "standard" or "cmaes_type"
        self.elite_weighting_type = config.get('weighting', "logarithmic")  # "uniform" or "logarithmic"
        self.noise_decay_factor = config.get('sigma_decay', 0.995)
        self.min_std_dev = config.get('min_sigma', 1e-3)
        self.epsilon = config.get('extra_noise_scale', 0.01)

        # 3. Initialize State
        self.num_elites = max(1, int(self.population_size * elite_fraction))
        self.std_devs = np.full(self.param_dim, initial_std_dev, dtype=np.float32)

        # For "cmaes_type" rule, need the mean used for sampling the current generation
        self.old_mean = np.copy(self.mean)

    @property
    def sigma(self):
        """Returns mean std dev for logging compatibility."""
        return np.mean(self.std_devs)

    def ask(self) -> np.ndarray:
        """
        Generates a population of candidate parameters.
        Renamed from 'sample_population' for main_train compatibility.
        """
        # Store the mean used for this generation's sampling if using cmaes_type rule
        if self.update_rule_type == "cmaes_type":
            self.old_mean = np.copy(self.mean)

        # Vectorized generation is faster than loop
        noise = np.random.randn(self.population_size, self.param_dim).astype(np.float32)
        population = self.mean + (self.std_devs * noise)

        return population

    def tell(self, candidates: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """
        Updates the distribution based on performance.
        Renamed from 'update_distribution' for main_train compatibility.
        """
        # Combine into list of tuples for internal logic
        evaluated_population = []
        for i in range(len(fitness_scores)):
            evaluated_population.append((candidates[i], fitness_scores[i]))

        # Sort by fitness (Descending: Higher score is better)
        evaluated_population.sort(key=lambda x: x[1], reverse=True)

        elite_individuals_params = [ind[0] for ind in evaluated_population[:self.num_elites]]
        elite_params_array = np.array(elite_individuals_params, dtype=np.float32)

        # Calculate elite weights (lambda_i)
        lambda_ = self._calculate_elite_weights()

        # 1. Update mean using weighted average of elites. Eq. 1 from the paper.
        self.mean = np.average(elite_params_array, axis=0, weights=lambda_)

        # 2. Update std_devs using Eq. 3
        if self.update_rule_type == "standard":
            # Variance around NEW mean
            squared_diffs = np.square(elite_params_array - self.mean)
        elif self.update_rule_type == "cmaes_type":
            # Variance around OLD mean (CMA-ES style approximation)
            squared_diffs = np.square(elite_params_array - self.old_mean)
        else:
            squared_diffs = np.square(elite_params_array - self.mean)

        # First, calculate the weighted average to get the new variance
        new_variances = np.average(squared_diffs, axis=0, weights=lambda_)

        # Then, add the noise to the final calculated variance
        self.std_devs = np.sqrt(new_variances + self.epsilon)

        # 3. Ensure std_devs do not collapse
        self.std_devs = np.maximum(self.std_devs, self.min_std_dev)

        # 4. Decay the extra noise scale
        self.epsilon *= self.noise_decay_factor
        self.epsilon = max(self.epsilon, 1e-5)  # Prevent epsilon from vanishing completely

        # Logging (Optional)
        # best_fitness = evaluated_population[0][1]
        # print(f"   >>> [CEM] Best: {best_fitness:.2f} | Sigma: {np.mean(self.std_devs):.4f}")

        return self.mean

    def _calculate_elite_weights(self) -> np.ndarray:
        """Calculates weights for elite individuals based on the configured type."""
        if self.elite_weighting_type == "logarithmic":
            # gives more importance to better individuals
            ranks = np.arange(1, self.num_elites + 1)
            raw_weights = np.log(self.num_elites + 1) - np.log(ranks)

            # Safety check
            if np.sum(raw_weights) <= 0:
                return np.full(self.num_elites, 1.0 / self.num_elites, dtype=np.float32)

            weights = raw_weights / np.sum(raw_weights)
            return weights.astype(np.float32)
        else:
            # "uniform": each individual is given the same importance
            return np.full(self.num_elites, 1.0 / self.num_elites, dtype=np.float32)

    def get_best_params(self) -> np.ndarray:
        return self.mean

    def boost_exploration(self, factor=1.5):
        """
        Called by CurriculumManager when the task gets harder.
        Increases variance to encourage finding new solutions.
        """
        old_mean_sigma = np.mean(self.std_devs)

        # Boost both the explicit std_devs and the epsilon noise floor
        self.std_devs = np.clip(self.std_devs * factor, a_min=None, a_max=2.0)
        self.epsilon *= factor

        print(f"   >>> [Optimizer] Exploration Boosted: {old_mean_sigma:.3f} -> {np.mean(self.std_devs):.3f}")
