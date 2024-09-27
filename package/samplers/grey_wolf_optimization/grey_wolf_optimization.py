from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers import RandomSampler
from optuna.study._study_direction import StudyDirection
import optunahub


class GreyWolfOptimizationSampler(optunahub.load_module("samplers/simple").SimpleBaseSampler):  # type: ignore
    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        population_size: int = 10,
        max_iter: int = 40,
        num_leaders: int = 3,
        seed: int = 0,
    ) -> None:
        # Initialize the base class
        super().__init__(search_space, seed)

        self.population_size = population_size
        self.max_iter = max_iter
        self.num_leaders = min(max(1, num_leaders), self.population_size // 2)
        self._rng = np.random.RandomState(seed)
        self.dim = 0
        self.leaders: np.ndarray = np.array([])  # Leaders (alpha, beta, gamma, ...) positions
        self.wolves: np.ndarray = np.array([])  # Wolf positions
        self.fitnesses: np.ndarray = np.full(population_size, np.inf)  # Fitness values
        self._random_sampler = RandomSampler(seed=seed)
        self.queue: list[dict[str, Any]] = []  # Queue to hold candidate positions

    def _lazy_init(self, search_space: dict[str, BaseDistribution]) -> None:
        # Workaround for the limitation of the type of distributions
        if any(
            isinstance(dist, optuna.distributions.CategoricalDistribution)
            for dist in search_space.values()
        ):
            raise NotImplementedError(
                "CategoricalDistribution is not supported in GreyWolfOptimizationSampler."
            )

        self.dim = len(search_space)
        self.lower_bound = np.array([dist.low for dist in search_space.values()])
        self.upper_bound = np.array([dist.high for dist in search_space.values()])
        self.wolves = (
            np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
            + self.lower_bound
        )
        self.leaders = np.zeros((self.num_leaders, self.dim))  # Initialize as zeros

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if len(search_space) == 0:
            return {}

        if self.dim != len(search_space):
            self._lazy_init(search_space)

        if len(self.queue) != 0:
            return self.queue.pop(0)

        if len(study.trials) < self.population_size:
            # Fill the initial population using sample_independent
            new_position = {
                k: self.sample_independent(study, trial, k, dist)
                for k, dist in search_space.items()
            }
            self.wolves[len(study.trials)] = np.array(list(new_position.values()))
            return new_position

        if len(study.trials) % self.population_size == 0:
            # Perform one iteration of GWO
            completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
            self.fitnesses = np.array(
                [trial.value for trial in completed_trials[-self.population_size :]]
            )
            self.fitnesses = np.array(
                [
                    (trial.value if study.direction == StudyDirection.MINIMIZE else -trial.value)
                    for trial in completed_trials[-self.population_size :]
                ]
            )

            # Update leaders (alpha, beta, gamma, ...)
            sorted_indices = np.argsort(self.fitnesses)
            self.leaders = self.wolves[sorted_indices[: self.num_leaders]]

            # Linearly decrease from 2 to 0
            a = 2 * (1 - len(study.trials) / (self.max_iter * self.population_size))

            # Calculate A, C, D, X values for position update
            r1 = self._rng.rand(self.population_size, self.num_leaders, self.dim)
            r2 = self._rng.rand(self.population_size, self.num_leaders, self.dim)
            A = 2 * a * r1 - a
            C = 2 * r2
            D = np.abs(C * self.leaders - self.wolves[:, np.newaxis, :])
            X = self.leaders - A * D

            # Update wolves' positions and clip to fit into the search space
            self.wolves = np.mean(X, axis=1)
            self.wolves = np.clip(self.wolves, self.lower_bound, self.upper_bound)

            # Store the wolves in the queue
            self.queue.extend(
                [{k: v for k, v in zip(search_space.keys(), pos)} for pos in self.wolves]
            )

        return self.queue.pop(0)

    def tell(self, new_positions: np.ndarray, fitnesses: np.ndarray) -> None:
        self.wolves = np.clip(new_positions, self.lower_bound, self.upper_bound)
        min_index = np.argmin(fitnesses)
        min_fitness = fitnesses[min_index]
        if min_fitness < self.fitnesses[min_index]:
            self.fitnesses[min_index] = min_fitness
            self.leaders[min_index] = self.wolves[min_index].copy()
