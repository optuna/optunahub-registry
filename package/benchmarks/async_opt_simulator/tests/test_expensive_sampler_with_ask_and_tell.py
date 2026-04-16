from __future__ import annotations

import time
from typing import Any
import unittest

import numpy as np
import optuna
import pytest

from .. import AsyncOptBenchmarkSimulator
from .utils import default_runtime_func
from .utils import get_configs
from .utils import ON_UBUNTU
from .utils import simplest_dummy_func
from .utils import TestProblem


class ExpensiveSampler(optuna.samplers.BaseSampler):
    """A sampler that sleeps proportionally to completed trials, returning values from a list."""

    def __init__(self, values: list[float], unittime: float):
        self._values = values
        self._unittime = unittime

    def infer_relative_search_space(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> dict:
        return {}

    def sample_relative(self, study: optuna.Study, trial: optuna.trial.FrozenTrial, search_space: dict) -> dict:
        return {}

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        n_completed = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
        if trial.number < len(self._values):
            time.sleep((n_completed + 1) * self._unittime)
            return self._values[trial.number]
        else:
            return float(10**5)


def optimize(index: int, n_workers: int):
    unittime = 1e-1 if ON_UBUNTU else 1.0
    configs, ans = get_configs(index=index, unittime=unittime)
    n_evals = configs.size

    sampler = ExpensiveSampler(values=configs.tolist(), unittime=unittime)
    search_space = {"x": optuna.distributions.FloatDistribution(0.0, float(10**6))}
    problem = TestProblem(obj_func=simplest_dummy_func, search_space=search_space)
    study = optuna.create_study(sampler=sampler)
    simulator = AsyncOptBenchmarkSimulator(
        n_workers=n_workers,
        allow_parallel_sampling=False,
    )
    simulator.optimize(study, problem, default_runtime_func, n_trials=n_evals)

    results = AsyncOptBenchmarkSimulator.get_results_from_study(study)
    diff = np.abs(np.array(results["cumtime"])[:n_evals] - ans)
    assert np.all(diff < unittime * 1.5)


@pytest.mark.parametrize("index", (0, 1, 2, 3, 4, 5, 6, 7, 8))
def test_opt(index: int) -> None:
    if index == 1:
        optimize(index=index, n_workers=2)
    elif ON_UBUNTU:
        optimize(index=index, n_workers=4)


if __name__ == "__main__":
    unittest.main()
