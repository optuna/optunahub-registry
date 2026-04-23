from __future__ import annotations

from typing import Any
import unittest

import numpy as np
import optuna
from optuna.trial import TrialState

from .. import AsyncOptBenchmarkSimulator
from .utils import CounterSampler
from .utils import default_runtime_func
from .utils import dummy_no_fidel_func
from .utils import get_overhead_from_study
from .utils import simplest_dummy_func
from .utils import TestProblem


DEFAULT_KWARGS = dict(
    n_workers=1,
    n_trials=10,
)

DUMMY_SEARCH_SPACE = {"x": optuna.distributions.IntDistribution(0, 99)}


def _create_study() -> optuna.Study:
    return optuna.create_study(sampler=CounterSampler())


def _create_problem(obj_func: Any = dummy_no_fidel_func) -> TestProblem:
    return TestProblem(obj_func=obj_func, search_space=DUMMY_SEARCH_SPACE)


# --- get_optimizer_overhead tests ---


def test_get_optimizer_overhead() -> None:
    """get_optimizer_overhead should return sampling time data with correct structure."""
    n_workers = DEFAULT_KWARGS["n_workers"]
    n_trials = DEFAULT_KWARGS["n_trials"]
    simulator = AsyncOptBenchmarkSimulator(n_workers=n_workers, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem()
    simulator.optimize(study, problem, default_runtime_func, n_trials=n_trials)

    overhead = get_overhead_from_study(study)
    assert "before_sample" in overhead
    assert "after_sample" in overhead
    assert "worker_id" in overhead
    # n_trials + n_workers - 1 ask() calls
    expected_len = n_trials + n_workers - 1
    assert len(overhead["before_sample"]) == expected_len
    assert len(overhead["after_sample"]) == expected_len
    assert len(overhead["worker_id"]) == expected_len
    # after_sample >= before_sample
    for before, after in zip(overhead["before_sample"], overhead["after_sample"]):
        assert after >= before


# --- property tests ---


def test_simulator_properties() -> None:
    """Public properties should return expected values."""
    n_workers = DEFAULT_KWARGS["n_workers"]
    simulator = AsyncOptBenchmarkSimulator(n_workers=n_workers, allow_parallel_sampling=False)
    assert simulator._n_workers == n_workers


# --- result ordering without parallel sampling ---


def test_results_sorted_by_cumtime_without_parallel_sampling() -> None:
    """Without parallel sampling, results should have cumtimes in non-decreasing order."""
    n_trials = 8
    simulator = AsyncOptBenchmarkSimulator(n_workers=4, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem(obj_func=simplest_dummy_func)
    simulator.optimize(study, problem, default_runtime_func, n_trials=n_trials)

    results = AsyncOptBenchmarkSimulator.get_results_from_study(study)
    cumtimes = np.array(results["cumtime"])
    assert np.all(cumtimes[:-1] <= cumtimes[1:])


def test_results_cumtime_monotonic_with_parallel_sampling() -> None:
    """With parallel sampling, results cumtimes should also be non-decreasing."""
    n_workers = DEFAULT_KWARGS["n_workers"]
    n_trials = DEFAULT_KWARGS["n_trials"]
    simulator = AsyncOptBenchmarkSimulator(n_workers=n_workers, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem()
    simulator.optimize(study, problem, default_runtime_func, n_trials=n_trials)

    results = AsyncOptBenchmarkSimulator.get_results_from_study(study)
    cumtimes = np.array(results["cumtime"])
    assert np.all(cumtimes[:-1] <= cumtimes[1:])


# --- tell_pending_result edge cases ---


def test_tell_skips_none_pending_results() -> None:
    """_tell_pending_result should skip workers with None pending results."""
    n_trials = DEFAULT_KWARGS["n_trials"]
    simulator = AsyncOptBenchmarkSimulator(n_workers=2, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem()
    simulator.optimize(study, problem, default_runtime_func, n_trials=n_trials)

    results = AsyncOptBenchmarkSimulator.get_results_from_study(study)
    assert len(results["cumtime"]) == n_trials


def test_multi_worker_all_results_collected() -> None:
    """With multiple workers, all n_trials results should be collected."""
    n_workers = 4
    n_trials = 12
    simulator = AsyncOptBenchmarkSimulator(n_workers=n_workers, allow_parallel_sampling=False)
    study = _create_study()
    problem = _create_problem()
    simulator.optimize(study, problem, default_runtime_func, n_trials=n_trials)

    results = AsyncOptBenchmarkSimulator.get_results_from_study(study)
    assert len(results["cumtime"]) == n_trials
    assert len(results["values"]) == n_trials
    assert len(results["worker_id"]) == n_trials


class _PruningProblem:
    """A problem where some trials get pruned based on the sampled parameter value."""

    search_space = {"x": optuna.distributions.IntDistribution(0, 99)}

    def __call__(self, trial: optuna.Trial) -> float:
        x = trial.suggest_int("x", 0, 99)
        trial.set_user_attr("runtime", 10.0)
        if x % 3 == 0:
            trial.report(float(x), step=0)
            raise optuna.TrialPruned(f"Pruned at x={x}")
        return float(x)


def test_simulator_handles_pruned_trials() -> None:
    """Simulator should complete successfully when some trials raise TrialPruned."""
    n_workers = 2
    n_trials = 10
    simulator = AsyncOptBenchmarkSimulator(n_workers=n_workers, allow_parallel_sampling=False)
    study = optuna.create_study(sampler=CounterSampler())
    problem = _PruningProblem()
    simulator.optimize(study, problem, default_runtime_func, n_trials=n_trials)

    results = AsyncOptBenchmarkSimulator.get_results_from_study(study)
    assert len(results["cumtime"]) == n_trials

    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    assert len(complete_trials) + len(pruned_trials) == n_trials
    assert len(pruned_trials) > 0, "At least some trials should be pruned"
    for t in complete_trials:
        assert t.values is not None


if __name__ == "__main__":
    unittest.main()
