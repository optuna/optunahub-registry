from __future__ import annotations

import time
from typing import Any

import numpy as np
import optuna
from optuna.samplers import RandomSampler
import optunahub


class DESampler(optunahub.samplers.SimpleBaseSampler):
    """Differential Evolution Sampler with Random Sampling for categorical parameters.

    This implements a hybrid sampling approach that:
    1. Uses DE algorithm for numerical parameters (float, int).
    2. Uses Random Sampling for categorical parameters.
    3. Combines both sampling strategies seamlessly.

    This also handles dynamic search space for numerical dimensions by:
        - For added dimensions in a trial:
            - Generation 0 (Random Sampling):
                The value for a new dimension is directly initialized by random sampling within the parameter's range.
            - Subsequent Generations (Differential Evolution):
                The new dimensions are initialized for the sampled individuals (r1, r2, r3) in the trial using the mean
                of the parameter's range. If the new dimension persists in subsequent trials, its values for the sampled
                individual in the trial are kept for subsequent trials.
        - For removed dimensions in a trial:
            Simply ignore the dimensions along with their values for all individuals.

    Args:
        search_space:
            Dictionary mapping parameter names to their distribution ranges.
        population_size:
            Number of individuals in the population.
        F:
            Mutation scaling factor - controls the amplification of differential evolution.
        CR:
            Crossover probability - controls the fraction of parameter values copied from the mutant.
        debug:
            Toggle for debug messages.
        seed:
            Random seed for reproducibility.

    Attributes:
        seed:
            Random seed for reproducibility.
        _rng:
            Random state object for sampling.
        _random_sampler:
            Random sampler instance for categorical parameters.
        population_size:
            Number of individuals in the population.
        F:
            Mutation scaling factor for DE.
        CR:
            Crossover probability for DE.
        debug:
            Debugging toggle.
        dim:
            Dimensionality of the search space.
        population:
            Population array for DE sampling.
        fitness:
            Fitness values of the population.
        trial_vectors:
            Trial vectors generated for a generation.
        lower_bound:
            Lower bounds of the numerical parameters.
        upper_bound:
            Upper bounds of the numerical parameters.
        numerical_params:
            List of numerical parameter names.
        categorical_params:
            List of categorical parameter names.
        last_time:
            Timestamp of the last performance measurement.
        last_trial_count:
            Count of trials completed at the last performance measurement.
        last_processed_gen:
            Last processed generation.
        current_gen_vectors:
            Trial vectors for the current generation.
    """

    def __init__(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution] | None = None,
        population_size: int | str = "auto",
        F: float = 0.8,
        CR: float = 0.7,
        debug: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize the DE sampler."""
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
        self.population: np.ndarray | None = None
        self.fitness: np.ndarray | None = None
        self.trial_vectors: np.ndarray | None = None
        self.lower_bound: np.ndarray | None = None
        self.upper_bound: np.ndarray | None = None

        # Parameter type tracking
        self.numerical_params: list[str] = []
        self.categorical_params: list[str] = []

        # Performance tracking
        self.last_time = time.time()
        self.last_trial_count = 0

        # Generation management
        self.last_processed_gen = -1
        self.current_gen_vectors: np.ndarray | None = None

        if self.population_size == "auto":
            self.population_size = self._determine_pop_size(search_space)

    def _determine_pop_size(
        self, search_space: dict[str, optuna.distributions.BaseDistribution] | None
    ) -> int:
        """Determine the population size based on the search space dimensionality.

        Args:
            search_space:
                Dictionary mapping parameter names to their distribution ranges.

        Returns:
            int:
                The population size.
        """
        if search_space is None:
            return 20
        else:
            dimension = len(search_space)

            # Start with a baseline multiplier
            if dimension < 5:
                # For very low dimension, maintain at least 20 individuals
                # to ensure diversity.
                base_multiplier = 10
                min_pop = 20
            elif dimension <= 30:
                # For moderately sized problems, a standard 10x dimension
                # is a good starting point.
                base_multiplier = 10
                min_pop = 30
            else:
                # For high-dimensional problems, start lower (5x)
                # to keep computations manageable.
                base_multiplier = 5
                min_pop = 50

            # Calculate a preliminary population size (can be fine-tuned further)
            population_size = max(min_pop, base_multiplier * dimension)

            return population_size

    def _split_search_space(
        self, search_space: dict[str, optuna.distributions.BaseDistribution]
    ) -> tuple[dict, dict]:
        """Split search space into numerical and categorical parameters.

        Args:
            search_space:
                Complete search space dictionary.

        Returns:
            tuple:
                A tuple of dictionaries (numerical_space, categorical_space).
        """
        numerical_space = {}
        categorical_space = {}

        for name, dist in search_space.items():
            if isinstance(
                dist,
                (optuna.distributions.FloatDistribution, optuna.distributions.IntDistribution),
            ):
                numerical_space[name] = dist
            else:
                categorical_space[name] = dist

        return numerical_space, categorical_space

    def _generate_trial_vectors(self, active_indices: list[int]) -> np.ndarray:
        """Generate new trial vectors using DE mutation and crossover.

        Args:
            active_indices:
                Indices of active dimensions in the current trial's search space.

        Returns:
            np.ndarray:
                Array of trial vectors (population_size x len(active_indices)).
        """

        if isinstance(self.population_size, str):
            raise ValueError("Population size must be resolved to an integer before this point.")

        trial_vectors = np.zeros((self.population_size, len(active_indices)))

        for i in range(self.population_size):
            # Select three random distinct individuals for mutation
            indices = [idx for idx in range(self.population_size) if idx != i]
            r1, r2, r3 = self._rng.choice(indices, 3, replace=False)

            if self.population is None or self.lower_bound is None or self.upper_bound is None:
                raise ValueError(
                    "Population, lower_bound, and upper_bound must be initialized before this operation."
                )

            # Handle NaN values by filling with default (mean of bounds)
            valid_population = np.nan_to_num(
                self.population[:, active_indices],
                nan=(self.lower_bound[active_indices] + self.upper_bound[active_indices]) / 2,
            )

            # Mutation: v = x_r1 + F * (x_r2 - x_r3) for active indices only
            mutant = valid_population[r1] + self.F * (valid_population[r2] - valid_population[r3])
            # Clip mutant vector to bounds for active dimensions
            mutant = np.clip(
                mutant, self.lower_bound[active_indices], self.upper_bound[active_indices]
            )

            # Crossover: combine target vector with mutant vector
            trial = np.copy(valid_population[i])
            crossover_mask = self._rng.rand(len(active_indices)) < self.CR

            # Ensure at least one parameter is taken from mutant vector
            if not np.any(crossover_mask):
                crossover_mask[self._rng.randint(len(active_indices))] = True
            trial[crossover_mask] = mutant[crossover_mask]

            trial_vectors[i] = trial

        return trial_vectors

    def _debug_print(self, message: str) -> None:
        """Print debug message if debug mode is enabled.

        Args:
            message:
                The message to print.
        """
        if self.debug:
            print(message)

    def _calculate_speed(self, n_completed: int) -> None:
        """Calculate and print optimization speed every 100 trials.

        Args:
            n_completed:
                The number of completed trials.
        """
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
        """Get all completed trials for a specific generation.

        Args:
            study:
                Optuna study object.
            generation:
                The generation number.

        Returns:
            list:
                A list of completed trials for the specified generation.
        """
        all_trials = study.get_trials(deepcopy=False)
        return [
            t
            for t in all_trials
            if (
                t.state == optuna.trial.TrialState.COMPLETE
                and t.system_attrs.get("differential_evolution:generation") == generation
            )
        ]

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        """Sample parameters for a trial using hybrid DE/Random sampling approach.

        Args:
            study:
                Optuna study object.
            trial:
                Current trial object.
            search_space:
                Dictionary of parameter distributions.

        Returns:
            dict:
                A dictionary of parameter values for the trial.
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

        # Track active dimensions for the current trial
        active_keys = list(numerical_space.keys())

        # Ensure numerical_params includes all possible keys
        if self.numerical_params is None:
            self.numerical_params = active_keys
        else:
            # Dynamically adjust numerical_params to reflect the current trial's search space
            for key in active_keys:
                if key not in self.numerical_params:
                    self.numerical_params.append(key)

        # Get indices for the active keys
        active_indices = [self.numerical_params.index(name) for name in active_keys]

        if not isinstance(self.population_size, int):
            raise ValueError(
                "Population size must be an integer before initializing trial vectors."
            )

        # Calculate current generation and individual index
        current_generation = trial._trial_id // self.population_size
        individual_index = trial._trial_id % self.population_size

        # Store generation and individual info as trial attributes
        study._storage.set_trial_system_attr(
            trial._trial_id, "differential_evolution:generation", current_generation
        )
        study._storage.set_trial_system_attr(
            trial._trial_id, "differential_evolution:individual", individual_index
        )

        self._calculate_speed(trial._trial_id)

        # Initialize population and bounds for the entire search space if not done
        if self.population is None:
            self._debug_print("\nInitializing population...")
            all_keys = list(numerical_space.keys())
            self.lower_bound = np.asarray([dist.low for dist in numerical_space.values()])
            self.upper_bound = np.asarray([dist.high for dist in numerical_space.values()])
            self.dim = len(all_keys)

            # Initialize population using seeded RNG
            self.population = (
                self._rng.rand(self.population_size, self.dim)
                * (self.upper_bound - self.lower_bound)
                + self.lower_bound
            )
            self.fitness = np.full(self.population_size, -np.inf if sign == -1 else np.inf)
            self.numerical_params = all_keys  # Track all keys

        # Initial population evaluation
        if current_generation == 0:
            self._debug_print(
                f"Evaluating initial individual {individual_index + 1}/{self.population_size}"
            )
            numerical_params = {
                name: (
                    float(value)
                    if isinstance(numerical_space[name], optuna.distributions.FloatDistribution)
                    else int(value)
                )
                for name, value in zip(
                    active_keys, self.population[individual_index, active_indices]
                )
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

                # Initialize trial_vectors with uniform size, using NaN or a default value for missing parameters
                trial_vectors = np.full(
                    (self.population_size, len(self.numerical_params)),
                    np.nan,  # Placeholder for missing parameters
                )

                for i, t in enumerate(prev_trials):
                    for j, name in enumerate(self.numerical_params):
                        if name in t.params:  # Only include active parameters
                            trial_vectors[i, j] = t.params[name]

                # if not isinstance(self.population_size, int):
                #     raise ValueError("Population size must be an integer before this point.")

                if self.fitness is None:
                    raise ValueError("Fitness array must be initialized before this operation.")

                if trial_fitness is None:
                    raise ValueError(
                        "Trial fitness array must be initialized before this operation."
                    )

                # Selection: keep better solutions
                for i in range(self.population_size):
                    if trial_fitness[i] <= sign * self.fitness[i]:
                        self.population[i, active_indices] = trial_vectors[i, active_indices]
                        self.fitness[i] = sign * trial_fitness[i]

                self._debug_print(f"Best fitness: {np.nanmin(sign * self.fitness):.6f}")

                # Generate new trial vectors for current generation
                self.current_gen_vectors = self._generate_trial_vectors(active_indices)
                self.last_processed_gen = current_generation

        # Ensure we have trial vectors for current generation
        if self.current_gen_vectors is None:
            self.current_gen_vectors = self._generate_trial_vectors(active_indices)

        # Combine numerical and categorical parameters
        numerical_params = {
            name: (
                float(value)
                if isinstance(numerical_space[name], optuna.distributions.FloatDistribution)
                else int(value)
            )
            for name, value in zip(active_keys, self.current_gen_vectors[individual_index])
        }
        return {**numerical_params, **categorical_params}
