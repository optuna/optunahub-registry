"""
Differential Evolution (DE) Sampler for Optuna.
This implements a DE algorithm that:
1. Initializes population randomly
2. For each generation:
   - Evaluates all individuals
   - For each individual:
     * Generates trial vector (mutation + crossover)
     * Evaluates trial vector
     * Keeps better solution (selection)
   - Generates new trial vectors for next generation
3. Repeats until stopping criterion met
"""

from __future__ import annotations

from typing import Any
import numpy as np
import optuna
import optunahub
import time


class DESampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(
            self,
            search_space: dict[str, optuna.distributions.BaseDistribution] | None = None,
            population_size: int = 50,
            F: float = 0.8,  # Mutation factor
            CR: float = 0.7,  # Crossover probability
            debug: bool = False,
    ) -> None:
        """Initialize the DE sampler.

        Args:
            search_space: Dictionary mapping parameter names to their distribution ranges
            population_size: Number of individuals in the population
            F: Mutation scaling factor - controls the amplification of differential evolution
            CR: Crossover probability - controls the fraction of parameter values copied from the mutant
            debug: Toggle for debug messages
        """
        super().__init__(search_space)

        # Initialize random number generator
        self._rng = np.random.RandomState()

        # DE algorithm parameters
        self.population_size = population_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover rate
        self.debug = debug

        # Search space parameters
        self.dim = 0  # Dimension of search space
        self.population = None  # Current population array (population_size x dim)
        self.fitness = None  # Fitness values of current population
        self.trial_vectors = None  # Trial vectors for mutation/crossover

        # Performance tracking
        self.last_time = time.time()  # For speed calculation
        self.last_trial_count = 0  # For speed calculation

        # Generation management
        self.last_processed_gen = -1  # Track last processed generation
        self.current_gen_vectors = None  # Trial vectors for current generation

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
        """Calculate and print optimization speed every 100 trials.

        Args:
            n_completed: Number of completed trials
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
            study: Optuna study object
            generation: Generation number to filter trials

        Returns:
            list: List of completed trials for the specified generation
        """
        # Get trials without deep copying for efficiency
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
        """Sample parameters for a trial using DE algorithm.

        This is the main method called by Optuna for each trial.

        Args:
            study: Optuna study object
            trial: Current trial object
            search_space: Dictionary of parameter distributions

        Returns:
            dict: Parameter values for the trial
        """
        if len(search_space) == 0:
            return {}

        # Use trial ID to determine generation and individual index
        current_generation = trial._trial_id // self.population_size
        individual_index = trial._trial_id % self.population_size

        # Store generation info in trial
        study._storage.set_trial_system_attr(trial._trial_id, "de:generation", current_generation)
        study._storage.set_trial_system_attr(trial._trial_id, "de:individual", individual_index)

        self._calculate_speed(trial._trial_id)

        # Initialize search space dimensions and bounds (first call only)
        if self.population is None:
            self._debug_print("\nInitializing population...")
            self.lower_bound = np.asarray([dist.low for dist in search_space.values()])
            self.upper_bound = np.asarray([dist.high for dist in search_space.values()])
            self.dim = len(search_space)
            # Initialize population randomly within bounds
            self.population = (
                    np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
                    + self.lower_bound
            )
            self.fitness = np.full(self.population_size, np.inf)
            study._storage.set_trial_system_attr(trial._trial_id, "de:dim", self.dim)

        # Initial population evaluation
        if current_generation == 0:
            self._debug_print(f"Evaluating initial individual {individual_index + 1}/{self.population_size}")
            return {k: v for k, v in zip(search_space.keys(), self.population[individual_index])}

        # Process previous generation if needed
        if current_generation != self.last_processed_gen:
            prev_gen = current_generation - 1
            prev_trials = self._get_generation_trials(study, prev_gen)

            if len(prev_trials) == self.population_size:
                self._debug_print(f"\nProcessing generation {prev_gen}")

                # Get fitness values and parameters from previous generation
                trial_fitness = np.array([t.value for t in prev_trials])
                trial_vectors = np.array([
                    [t.params[f"x{i}"] for i in range(self.dim)]
                    for t in prev_trials
                ])

                # Selection: keep better solutions
                for i in range(self.population_size):
                    if trial_fitness[i] <= self.fitness[i]:
                        self.population[i] = trial_vectors[i]
                        self.fitness[i] = trial_fitness[i]

                self._debug_print(f"Best fitness: {np.min(self.fitness):.6f}")

                # Generate new trial vectors for current generation
                self.current_gen_vectors = self._generate_trial_vectors()
                self.last_processed_gen = current_generation

        # Generate trial vectors if needed
        if self.current_gen_vectors is None:
            self.current_gen_vectors = self._generate_trial_vectors()

        # Return parameters for current individual
        return {k: v for k, v in zip(search_space.keys(), self.current_gen_vectors[individual_index])}