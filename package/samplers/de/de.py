from __future__ import annotations

from typing import Any
import numpy as np
import optuna
import optunahub


class DESampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution] | None = None,
        population_size: int = 50,
        F: float = 0.8,  # Mutation factor
        CR: float = 0.7,  # Crossover probability
    ) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.queue: list[dict[str, Any]] = []  # Stores individuals as parameter dictionaries
        self.dim = 0  # Will represent the dimension of the search space
        self.population = None  # Population array
        self.fitness = None  # Array to store fitness values

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:

        if len(search_space) == 0:
            return {}
        if len(self.queue) != 0:
            return self.queue.pop(0)

        # Initialize search space dimensions and bounds
        if self.population is None:
            self.lower_bound = np.asarray([dist.low for dist in search_space.values()])
            self.upper_bound = np.asarray([dist.high for dist in search_space.values()])
            self.dim = len(search_space)
            self.population = (
                np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
                + self.lower_bound
            )
            self.fitness = np.full(self.population_size, np.inf)

        # Evaluate fitness of individuals
        last_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))[
            -self.population_size :
        ]
        for i, trial in enumerate(last_trials):
            self.fitness[i] = trial.value if trial.value is not None else self.fitness[i]

        new_population = np.zeros_like(self.population)

        for i in range(self.population_size):
            # Mutation: Generate a mutant vector
            indices = [idx for idx in range(self.population_size) if idx != i]
            r1, r2, r3 = self._rng.choice(indices, 3, replace=False)
            mutant = (
                self.population[r1]
                + self.F * (self.population[r2] - self.population[r3])
            )
            mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

            # Crossover: Create a trial vector
            trial = np.copy(self.population[i])
            crossover_mask = self._rng.rand(self.dim) < self.CR
            if not np.any(crossover_mask):
                crossover_mask[self._rng.randint(self.dim)] = True
            trial[crossover_mask] = mutant[crossover_mask]

            # Print vectors for debugging
            print(f"Target Vector (Individual {i}): {self.population[i]}")
            print(f"Mutant Vector: {mutant}")
            print(f"Resultant Vector (After Crossover): {trial}")

            # Add trial vector to new population
            new_population[i] = trial

        # Convert new population into a parameter list for the queue
        param_list = [
            {k: v for k, v in zip(search_space.keys(), individual)}
            for individual in new_population
        ]

        self.queue.extend(param_list)

        return self.queue.pop(0)
