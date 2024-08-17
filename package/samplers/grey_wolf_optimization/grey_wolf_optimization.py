from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import optunahub
from optuna.distributions import BaseDistribution
from optuna.samplers import RandomSampler


class GreyWolfOptimizationSampler(optunahub.load_module("samplers/simple").SimpleBaseSampler):  # type: ignore
    @dataclass
    class Wolf:
        position: np.ndarray
        fitness: float

    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        population_size: int = 10,
        max_iter: int = 40,
        num_leaders: int = 3,
        seed: int = 0,
    ) -> None:
        self.search_space = search_space
        self.population_size = population_size
        self.max_iter = max_iter
        self.num_leaders = min(max(1, num_leaders), self.population_size // 2)
        self._rng = np.random.RandomState(seed)
        self.dim = 0
        self.leaders: list[self.Wolf] = []  # Leaders (alpha, beta, gamma, ...)
        self.wolves: list[self.Wolf] = []
        self._random_sampler = RandomSampler(seed=seed)

    def _lazy_init(self, search_space: dict[str, BaseDistribution]) -> None:
        self.dim = len(search_space)
        self.lower_bound = np.array([dist.low for dist in search_space.values()])
        self.upper_bound = np.array([dist.high for dist in search_space.values()])
        self.wolves = [
            self.Wolf(
                position=np.random.rand(self.dim)
                * (self.upper_bound - self.lower_bound)
                + self.lower_bound,
                fitness=np.inf,
            )
            for _ in range(self.population_size)
        ]
        self.leaders = [None] * self.num_leaders  # Initialize as None

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

        if len(study.trials) < self.population_size:
            # Fill the initial population using sample_independent
            new_position = {
                k: self.sample_independent(study, trial, k, dist)
                for k, dist in search_space.items()
            }
            self.wolves[len(study.trials)] = self.Wolf(
                position=np.array(list(new_position.values())),
                fitness=np.inf,
            )
            return new_position

        if len(study.trials) % self.population_size == 0:
            # Perform one iteration of GWO
            completed_trials = study.get_trials(
                states=[optuna.trial.TrialState.COMPLETE]
            )
            fitnesses = np.array(
                [trial.value for trial in completed_trials[-self.population_size :]]
            )

            # Update leaders (alpha, beta, gamma, ...)
            sorted_indices = np.argsort(fitnesses)
            self.leaders = [
                self.wolves[sorted_indices[i]] for i in range(self.num_leaders)
            ]

            # Linearly decrease from 2 to 0
            a = 2 * (1 - len(study.trials) / (self.max_iter * self.population_size))

            for i in range(self.population_size):
                A = [
                    a * (2 * self._rng.rand(self.dim) - 1)
                    for _ in range(self.num_leaders)
                ]  # Coefficients A1, A2, A3, ...
                C = [
                    2 * self._rng.rand(self.dim) for _ in range(self.num_leaders)
                ]  # Coefficients C1, C2, C3, ...

                D = [
                    abs(C[j] * self.leaders[j].position - self.wolves[i].position)
                    for j in range(self.num_leaders)
                ]

                X = [
                    self.leaders[j].position - A[j] * D[j]
                    for j in range(self.num_leaders)
                ]

                self.wolves[i].position = np.mean(X, axis=0)

        next_wolf_position = self.wolves[
            len(study.trials) % self.population_size
        ].position
        return {k: v for k, v in zip(search_space.keys(), next_wolf_position)}


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    sampler = GreyWolfOptimizationSampler(
        {
            "x": optuna.distributions.FloatDistribution(-10, 10),
            "y": optuna.distributions.FloatDistribution(-10, 10),
        }
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=100)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()
