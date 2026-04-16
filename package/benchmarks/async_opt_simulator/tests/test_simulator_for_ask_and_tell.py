from __future__ import annotations

import unittest

import numpy as np
import optuna
import pytest

from .. import AsyncOptBenchmarkSimulator
from .utils import CounterSampler
from .utils import default_runtime_func
from .utils import dummy_no_fidel_func
from .utils import TestProblem


DEFAULT_KWARGS = dict(
    n_workers=1,
    n_trials=10,
)

DUMMY_SEARCH_SPACE = {"x": optuna.distributions.IntDistribution(0, 99)}


def _create_study() -> optuna.Study:
    return optuna.create_study(sampler=CounterSampler())


def _create_problem() -> TestProblem:
    return TestProblem(obj_func=dummy_no_fidel_func, search_space=DUMMY_SEARCH_SPACE)


def test_proc_obj_func_works():
    """_proc_obj_func works without fidels."""
    simulator = AsyncOptBenchmarkSimulator(n_workers=1, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem()
    trial = study.ask(problem.search_space)
    simulator._proc_obj_func(trial=trial, problem=problem, runtime_func=default_runtime_func, worker_id=0)


def test_results_monotonically_ordered() -> None:
    """Results should be reported in non-decreasing cumtime order."""
    n_workers = 4
    n_trials = 10
    simulator = AsyncOptBenchmarkSimulator(n_workers=n_workers, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem()
    simulator.optimize(study, problem, default_runtime_func, n_trials=n_trials)

    results = AsyncOptBenchmarkSimulator.get_results_from_study(study)
    cumtimes = np.array(results["cumtime"])
    assert len(cumtimes) == n_trials
    assert np.allclose(np.maximum.accumulate(cumtimes), cumtimes)


def test_error_missing_runtime():
    """Problem must set runtime user_attr."""
    simulator = AsyncOptBenchmarkSimulator(n_workers=1, allow_parallel_sampling=False)
    study = optuna.create_study(sampler=CounterSampler())

    class BadProblem:
        search_space = {"x": optuna.distributions.IntDistribution(0, 9)}

        def __call__(self, trial: optuna.Trial) -> float:
            return 0.0

    problem = BadProblem()
    trial = study.ask(problem.search_space)
    with pytest.raises(KeyError, match="runtime"):
        simulator._proc_obj_func(trial=trial, problem=problem, runtime_func=default_runtime_func, worker_id=0)


if __name__ == "__main__":
    unittest.main()
