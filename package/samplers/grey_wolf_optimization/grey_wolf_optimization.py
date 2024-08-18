from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import optunahub
from optuna.distributions import BaseDistribution
from optuna.samplers import RandomSampler


class GreyWolfOptimizationSampler(optunahub.load_module("samplers/simple").SimpleBaseSampler):  # type: ignore
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
        self.leaders: np.ndarray = np.array(
            []
        )  # Leaders (alpha, beta, gamma, ...) positions
        self.wolves: np.ndarray = np.array([])  # Wolf positions
        self.fitnesses: np.ndarray = np.full(population_size, np.inf)  # Fitness values
        self._random_sampler = RandomSampler(seed=seed)

    def _lazy_init(self, search_space: dict[str, BaseDistribution]) -> None:
        self.dim = len(search_space)
        self.lower_bound = np.array([dist.low for dist in search_space.values()])
        self.upper_bound = np.array([dist.high for dist in search_space.values()])
        self.wolves = (
            np.random.rand(self.population_size, self.dim)
            * (self.upper_bound - self.lower_bound)
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
            completed_trials = study.get_trials(
                states=[optuna.trial.TrialState.COMPLETE]
            )
            self.fitnesses = np.array(
                [trial.value for trial in completed_trials[-self.population_size :]]
            )

            # Update leaders (alpha, beta, gamma, ...)
            sorted_indices = np.argsort(self.fitnesses)
            self.leaders = self.wolves[sorted_indices[: self.num_leaders]]

            # Linearly decrease from 2 to 0
            a = 2 * (1 - len(study.trials) / (self.max_iter * self.population_size))

            for i in range(self.population_size):
                r1 = self._rng.rand(self.num_leaders, self.dim)
                r2 = self._rng.rand(self.num_leaders, self.dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.leaders - self.wolves[i])
                X = np.mean(self.leaders - A * D, axis=0)

                self.wolves[i] = np.clip(X, self.lower_bound, self.upper_bound)

        next_wolf_position = self.wolves[len(study.trials) % self.population_size]

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
        },
        num_leaders=5,
    )
    study = optuna.create_study(sampler=sampler, direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=100)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()
