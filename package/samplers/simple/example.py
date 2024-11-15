from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial import FrozenTrial
import optunahub


class UserDefinedSampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(self, search_space: dict[str, BaseDistribution] | None = None) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        params = {}
        for n, d in search_space.items():
            if isinstance(d, FloatDistribution):
                params[n] = self._rng.uniform(d.low, d.high)
            elif isinstance(d, IntDistribution):
                params[n] = self._rng.randint(d.low, d.high)
            else:
                raise ValueError("Unsupported distribution")
        return params


if __name__ == "__main__":

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0, 1)

        return x

    sampler = UserDefinedSampler({"x": FloatDistribution(0, 1)})
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)

    print(study.best_trial.value, study.best_trial.params)
