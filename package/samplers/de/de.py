from __future__ import annotations

from typing import Any
import numpy as np
import optuna
from optuna.samplers import RandomSampler
import optunahub
import time


class DESampler(optunahub.samplers.SimpleBaseSampler):
    """Differential Evolution Sampler with Random Sampling for categorical parameters.

    This implements a hybrid sampling approach that:
    1. Uses DE algorithm for numerical parameters (float, int)
    2. Uses Random Sampling for categorical parameters
    3. Combines both sampling strategies seamlessly
    """

    def __init__(
            self,
            search_space: dict[str, optuna.distributions.BaseDistribution] | None = None,
            population_size: int = 50,
            F: float = 0.8,  # Mutation factor
            CR: float = 0.7,  # Crossover probability
            debug: bool = False,
            seed: int | None = None,  # Random seed
    ) -> None:
        """Initialize the DE sampler.

        Args:
            search_space: Dictionary mapping parameter names to their distribution ranges
            population_size: Number of individuals in the population
            F: Mutation scaling factor - controls the amplification of differential evolution
            CR: Crossover probability - controls the fraction of parameter values copied from the mutant
            debug: Toggle for debug messages
            seed: Random seed for reproducibility
        """
        super().__init__(search_space)

        # Store and set random seed
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Initialize random sampler for categorical parameters
        self._random_sampler = RandomSampler(seed=seed)

        # DE algorithm parameters
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.debug = debug

        # Search space parameters
        self.dim = 0
        self.population = None
        self.fitness = None
        self.trial_vectors = None
        self.lower_bound = None
        self.upper_bound = None

        # Parameter type tracking
        self.numerical_params = []
        self.categorical_params = []

        # Performance tracking
        self.last_time = time.time()
        self.last_trial_count = 0

        # Generation management
        self.last_processed_gen = -1
        self.current_gen_vectors = None

    def _split_search_space(self, search_space: dict[str, optuna.distributions.BaseDistribution]) -> tuple[dict, dict]:
        """Split search space into numerical and categorical parameters.

        Args:
            search_space: Complete search space dictionary

        Returns:
            tuple: (numerical_space, categorical_space)
        """
        numerical_space = {}
        categorical_space = {}

        for name, dist in search_space.items():
            if isinstance(dist, (optuna.distributions.FloatDistribution,
                                 optuna.distributions.IntDistribution)):
                numerical_space[name] = dist
            else:
                categorical_space[name] = dist

        return numerical_space, categorical_space

    def _generate_trial_vectors(self) -> np.ndarray:
        """Generate new trial vectors using DE mutation and crossover.

        Returns:
            np.ndarray: Array of trial vectors (population_size x dim)
        """
        trial_vectors = np.zeros_like(self.population)

        for i in range(self.population_size):
            # Select three random distinct individuals for mutation
            indices = [idx for idx in range(self.population_size) if idx != i]
            r1, r2, r3 = self._rng.choice(indices, 3, replace=False)

            # Mutation: v = x_r1 + F * (x_r2 - x_r3)
            mutant = (
                    self.population[r1]
                    + self.F * (self.population[r2] - self.population[r3])
            )
            # Clip mutant vector to bounds
            mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

            # Crossover: combine target vector with mutant vector
            trial = np.copy(self.population[i])
            # Generate crossover mask based on CR
            crossover_mask = self._rng.rand(self.dim) < self.CR
            # Ensure at least one parameter is taken from mutant vector
            if not np.any(crossover_mask):
                crossover_mask[self._rng.randint(self.dim)] = True
            trial[crossover_mask] = mutant[crossover_mask]

            trial_vectors[i] = trial

        return trial_vectors

    def _debug_print(self, message: str) -> None:
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(message)

    def _calculate_speed(self, n_completed: int) -> None:
        """Calculate and print optimization speed every 100 trials."""
        if not self.debug:
            return

        if n_completed % 100 == 0 and n_completed > 0:
            current_time = time.time()
            elapsed_time = current_time - self.last_time
            trials_processed = n_completed - self.last_trial_count

            if elapsed_time > 0:
                speed = trials_processed / elapsed_time
                print(f"\n[Speed Stats] Trials {self.last_trial_count} to {n_completed}")
                print(f"Speed: {speed:.2f} trials/second")
                print(f"Time elapsed: {elapsed_time:.2f} seconds")
                print("-" * 50)

            self.last_time = current_time
            self.last_trial_count = n_completed

    def _get_generation_trials(self, study: optuna.study.Study, generation: int) -> list:
        """Get all completed trials for a specific generation."""
        all_trials = study.get_trials(deepcopy=False)
        return [
            t for t in all_trials
            if (t.state == optuna.trial.TrialState.COMPLETE and
                t.system_attrs.get("de:generation") == generation)
        ]

    def sample_relative(
            self,
            study: optuna.study.Study,
            trial: optuna.trial.FrozenTrial,
            search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        """Sample parameters for a trial using hybrid DE/Random sampling approach.

        Args:
            study: Optuna study object
            trial: Current trial object
            search_space: Dictionary of parameter distributions

        Returns:
            dict: Parameter values for the trial
        """
        if len(search_space) == 0:
            return {}

        # Determine the direction of optimization
        sign = 1 if study.direction == optuna.study.StudyDirection.MINIMIZE else -1

        # Split search space into numerical and categorical
        numerical_space, categorical_space = self._split_search_space(search_space)

        # Sample categorical parameters using random sampler
        categorical_params = {}
        for param_name, distribution in categorical_space.items():
            categorical_params[param_name] = self._random_sampler.sample_independent(
                study, trial, param_name, distribution
            )

        # If no numerical parameters, return only categorical
        if not numerical_space:
            return categorical_params

        # Calculate current generation and individual index
        current_generation = trial._trial_id // self.population_size
        individual_index = trial._trial_id % self.population_size

        # Store generation and individual info as trial attributes
        study._storage.set_trial_system_attr(trial._trial_id, "de:generation", current_generation)
        study._storage.set_trial_system_attr(trial._trial_id, "de:individual", individual_index)

        self._calculate_speed(trial._trial_id)

        # Initialize search space dimensions and bounds if not done
        if self.population is None:
            self._debug_print("\nInitializing population...")
            self.lower_bound = np.asarray([dist.low for dist in numerical_space.values()])
            self.upper_bound = np.asarray([dist.high for dist in numerical_space.values()])
            self.dim = len(numerical_space)
            # Initialize population using seeded RNG
            self.population = (
                    self._rng.rand(self.population_size, self.dim) *
                    (self.upper_bound - self.lower_bound) +
                    self.lower_bound
            )
            # Initialize fitness based on direction
            self.fitness = np.full(self.population_size , -np.inf if sign == -1 else np.inf)
            study._storage.set_trial_system_attr(trial._trial_id, "de:dim", self.dim)
            # Store parameter names for later reference
            self.numerical_params = list(numerical_space.keys())

        # Initial population evaluation
        if current_generation == 0:
            self._debug_print(f"Evaluating initial individual {individual_index + 1}/{self.population_size}")
            numerical_params = {
                name: (float(value) if isinstance(numerical_space[name],
                                                  optuna.distributions.FloatDistribution) else int(value))
                for name, value in zip(self.numerical_params, self.population[individual_index])
            }
            return {**numerical_params, **categorical_params}

        # Process previous generation if needed
        if current_generation > 0 and current_generation != self.last_processed_gen:
            prev_gen = current_generation - 1
            prev_trials = self._get_generation_trials(study, prev_gen)

            if len(prev_trials) == self.population_size:
                self._debug_print(f"\nProcessing generation {prev_gen}")

                # Get fitness and parameter values from previous generation
                trial_fitness = np.array([sign * t.value for t in prev_trials])
                trial_vectors = np.array([
                    [t.params[name] for name in self.numerical_params]
                    for t in prev_trials
                ])

                # Selection: keep better solutions
                for i in range(self.population_size):
                    if trial_fitness[i] <= sign * self.fitness[i]:
                        self.population[i] = trial_vectors[i]
                        self.fitness[i] = sign * trial_fitness[i]

                self._debug_print(f"Best fitness: {np.min(sign * self.fitness):.6f}")

                # Generate new trial vectors for current generation
                self.current_gen_vectors = self._generate_trial_vectors()
                self.last_processed_gen = current_generation

        # Ensure we have trial vectors for current generation
        if self.current_gen_vectors is None:
            self.current_gen_vectors = self._generate_trial_vectors()

        # Combine numerical and categorical parameters
        numerical_params = {
            name: (float(value) if isinstance(numerical_space[name],
                                              optuna.distributions.FloatDistribution) else int(value))
            for name, value in zip(self.numerical_params, self.current_gen_vectors[individual_index])
        }
        return {**numerical_params, **categorical_params}
