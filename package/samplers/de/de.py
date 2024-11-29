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
            CR: float = 0.7,  # Crossover probability,
            debug: bool = False,  # Toggle for debug messages
    ) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.debug = debug
        self.queue: list[dict[str, Any]] = []  # Stores individuals as parameter dictionaries
        self.dim = 0  # Will represent the dimension of the search space
        self.population = None  # Population array
        self.fitness = None  # Array to store fitness values
        self.trial_vectors = None  # Store trial vectors for selection
        self.current_generation = 0  # Track current generation
        self.pending_evaluations = 0  # Track pending evaluations in current generation
        self.last_processed_trial = 0  # Track the last processed trial

        # Speed tracking variables
        self.last_time = time.time()
        self.last_trial_count = 0

    def _debug_print(self, message: str) -> None:
        """Helper method for debug printing"""
        if self.debug:
            print(message)

    def _calculate_speed(self, n_completed: int) -> None:
        """Calculate and print the speed for recent trials"""
        if not self.debug:
            return

        if n_completed % 100 == 0 and n_completed > 0:
            current_time = time.time()
            elapsed_time = current_time - self.last_time
            trials_processed = n_completed - self.last_trial_count

            if elapsed_time > 0:  # Avoid division by zero
                speed = trials_processed / elapsed_time
                print(f"\n[Speed Stats] Trials {self.last_trial_count} to {n_completed}")
                print(f"Speed: {speed:.2f} trials/second")
                print(f"Time elapsed: {elapsed_time:.2f} seconds")
                print("-" * 50)

            # Update tracking variables
            self.last_time = current_time
            self.last_trial_count = n_completed

    def sample_relative(
            self,
            study: optuna.study.Study,
            trial: optuna.trial.FrozenTrial,
            search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        if len(search_space) == 0:
            return {}

        completed_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        n_completed = len(completed_trials)
        self._calculate_speed(n_completed)  # Calculate speed statistics
        self._debug_print(f"\nTotal completed trials: {n_completed}")

        # Initialize search space dimensions and bounds
        if self.population is None:
            self._debug_print("\nInitializing population...")
            self.lower_bound = np.asarray([dist.low for dist in search_space.values()])
            self.upper_bound = np.asarray([dist.high for dist in search_space.values()])
            self.dim = len(search_space)
            self.population = (
                    np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
                    + self.lower_bound
            )
            self.fitness = np.full(self.population_size, np.inf)
            self.trial_vectors = np.zeros_like(self.population)

            # Initial population evaluation
        if self.current_generation == 0:
            if n_completed < self.population_size:
                self._debug_print(f"Evaluating initial individual {n_completed + 1}/{self.population_size}")
                params = {
                    k: v for k, v in zip(search_space.keys(), self.population[n_completed])
                }
                return params
            else:
                self._debug_print("\nInitial population evaluation complete.")
                self._debug_print("Starting evolution process...")
                for i in range(self.population_size):
                    self.fitness[i] = completed_trials[i].value
                self.current_generation = 1
                self.pending_evaluations = 0

            # Check if we need to process the previous generation's results
        if self.pending_evaluations >= self.population_size and n_completed >= (
                self.current_generation * self.population_size):
            self._debug_print(f"\nGeneration {self.current_generation} Selection:")
            self._debug_print("-" * 50)

            # Get the fitness values for the trial vectors from the most recent trials
            start_idx = n_completed - self.population_size
            recent_trials = completed_trials[start_idx:n_completed]
            trial_fitness = np.array([trial.value for trial in recent_trials])

            # Selection: compare each trial vector with its target vector
            for i in range(self.population_size):
                self._debug_print(f"\nIndividual {i + 1}:")
                self._debug_print(f"Target Vector Fitness: {self.fitness[i]:.6f}")
                self._debug_print(f"Trial Vector Fitness:  {trial_fitness[i]:.6f}")

                if trial_fitness[i] <= self.fitness[i]:  # Minimization problem
                    self.population[i] = self.trial_vectors[i]
                    self.fitness[i] = trial_fitness[i]
                    self._debug_print("=> Selected: Trial Vector (Better)")
                else:
                    self._debug_print("=> Selected: Target Vector (Better)")

            self._debug_print(f"\nBest fitness in generation {self.current_generation}: {np.min(self.fitness):.6f}")
            self._debug_print("-" * 50)

            self.current_generation += 1
            self.pending_evaluations = 0
            self.last_processed_trial = n_completed
            self.queue.clear()  # Clear the queue before generating new trial vectors

            # Generate new trial vectors if queue is empty
        if len(self.queue) == 0:
            self._debug_print(f"\nGenerating trial vectors for generation {self.current_generation}")
            for i in range(self.population_size):
                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                r1, r2, r3 = self._rng.choice(indices, 3, replace=False)
                mutant = (
                        self.population[r1]
                        + self.F * (self.population[r2] - self.population[r3])
                )
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.copy(self.population[i])
                crossover_mask = self._rng.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[self._rng.randint(self.dim)] = True
                trial[crossover_mask] = mutant[crossover_mask]

                self.trial_vectors[i] = trial

            # Convert trial vectors to parameter dictionaries
            param_list = [
                {k: v for k, v in zip(search_space.keys(), individual)}
                for individual in self.trial_vectors
            ]
            self.queue.extend(param_list)

        self.pending_evaluations += 1
        self._debug_print(
            f"Evaluating individual {self.pending_evaluations}/{self.population_size} in generation {self.current_generation}")

        return self.queue.pop(0)