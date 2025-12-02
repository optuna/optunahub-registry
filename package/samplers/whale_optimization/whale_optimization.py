from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np
import optuna
from optuna.samplers._lazy_random_state import LazyRandomState
import optunahub


class WhaleOptimizationSampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(
        self,
        search_space: dict[str, optuna.distributions.BaseDistribution] | None = None,
        population_size: int = 10,
        max_iter: int = 40,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(search_space, seed)
        self._rng = LazyRandomState(seed)
        self.population_size = population_size
        self.max_iter = max_iter
        self.dim = 0
        self.queue: list[dict[str, Any]] = []

    def _lazy_init(self, search_space: dict[str, optuna.distributions.BaseDistribution]) -> None:
        assert all(
            isinstance(dist, optuna.distributions.FloatDistribution)
            for dist in search_space.values()
        )

        self.lower_bound = np.asarray([dist.low for dist in search_space.values()])
        self.upper_bound = np.asarray([dist.high for dist in search_space.values()])
        self.dim = len(search_space)
        self.leader_pos = (
            self._rng.rng.rand(self.dim) * (self.upper_bound - self.lower_bound) + self.lower_bound
        )
        self.leader_score = np.inf
        self.positions = (
            self._rng.rng.rand(self.population_size, self.dim)
            * (self.upper_bound - self.lower_bound)
            + self.lower_bound
        )

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        if len(search_space) == 0:
            return {}
        if self.dim != len(search_space):
            self._lazy_init(search_space)
        if len(self.queue) != 0:
            return self.queue.pop(0)
        last_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))[
            -self.population_size :
        ]
        current_iter = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
        new_positions = np.asarray([list(e.params.values()) for e in last_trials])
        fitnesses = np.asarray([e.value for e in last_trials])
        if current_iter > self.population_size:
            self.tell(new_positions, fitnesses)
        a = 2 - current_iter * (2 / self.max_iter)
        a2 = -1 + current_iter * (-1 / self.max_iter)
        new_positions = np.zeros_like(self.positions)

        for i in range(self.positions.shape[0]):
            r1, r2 = self._rng.rng.rand(), self._rng.rng.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            b, L = 1, ((a2 - 1) * self._rng.rng.rand() + 1)
            p = self._rng.rng.rand()
            if p < 0.5:
                if np.abs(A) >= 1:
                    rand_leader_index = self._rng.rng.randint(self.population_size)
                    X_rand = self.positions[rand_leader_index, :]
                    D_X_rand = np.abs(C * X_rand - self.positions[i, :])
                    new_positions[i, :] = X_rand - A * D_X_rand
                else:
                    D_Leader = np.abs(C * self.leader_pos - self.positions[i, :])
                    new_positions[i, :] = self.leader_pos - A * D_Leader
            else:
                distance2Leader = np.abs(self.leader_pos - self.positions[i, :])
                new_positions[i, :] = (
                    distance2Leader * np.exp(b * L) * np.cos(L * 2 * np.pi) + self.leader_pos
                )

        param_list = [
            {k: v for k, v in zip(search_space.keys(), new_pos)} for new_pos in new_positions
        ]
        self.queue.extend(param_list)
        return self.queue.pop(0)

    def tell(self, new_positions: np.ndarray, fitnesses: np.ndarray) -> None:
        self.positions = np.clip(new_positions, self.lower_bound, self.upper_bound)
        min_index = np.argmin(fitnesses)
        min_fitness = fitnesses[min_index]
        if min_fitness < self.leader_score:
            self.leader_score = min_fitness
            self.leader_pos = self.positions[min_index].copy()
