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
            F: float = 0.8,
            CR: float = 0.7,
            debug: bool = False,
    ) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.debug = debug
        self.dim = 0
        self.population = None
        self.fitness = None
        self.trial_vectors = None
        self.last_time = time.time()
        self.last_trial_count = 0
        self.last_processed_gen = -1
        self.current_gen_vectors = None  # Store vectors for current generation

    def _generate_trial_vectors(self) -> np.ndarray:
        """Generate new trial vectors using DE mutation and crossover."""
        trial_vectors = np.zeros_like(self.population)
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

            trial_vectors[i] = trial
        return trial_vectors

    def _debug_print(self, message: str) -> None:
        if self.debug:
            print(message)

    def _calculate_speed(self, n_completed: int) -> None:
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
        """Get trials for a specific generation using trial system attributes."""
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
        if len(search_space) == 0:
            return {}

        # Use trial_id for position tracking
        current_generation = trial._trial_id // self.population_size
        individual_index = trial._trial_id % self.population_size

        study._storage.set_trial_system_attr(trial._trial_id, "de:generation", current_generation)
        study._storage.set_trial_system_attr(trial._trial_id, "de:individual", individual_index)

        self._calculate_speed(trial._trial_id)

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

                # Get fitness values and parameters
                trial_fitness = np.array([t.value for t in prev_trials])
                trial_vectors = np.array([
                    [t.params[f"x{i}"] for i in range(self.dim)]
                    for t in prev_trials
                ])

                # Selection
                for i in range(self.population_size):
                    if trial_fitness[i] <= self.fitness[i]:
                        self.population[i] = trial_vectors[i]
                        self.fitness[i] = trial_fitness[i]

                self._debug_print(f"Best fitness: {np.min(self.fitness):.6f}")

                # Generate new trial vectors for current generation
                self.current_gen_vectors = self._generate_trial_vectors()
                self.last_processed_gen = current_generation

        # If we haven't generated trial vectors for this generation yet, do it now
        if self.current_gen_vectors is None:
            self.current_gen_vectors = self._generate_trial_vectors()

        # Return parameters for current individual
        return {k: v for k, v in zip(search_space.keys(), self.current_gen_vectors[individual_index])}