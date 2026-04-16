from __future__ import annotations

import unittest

from benchmark_apis import MFBranin
import numpy as np
import optuna

from .. import AsyncOptBenchmarkSimulator
from .utils import default_runtime_func


class MFBraninProblem:
    def __init__(self, bench: MFBranin):
        self.search_space = {
            "x0": optuna.distributions.FloatDistribution(0.0, 1.0),
            "x1": optuna.distributions.FloatDistribution(0.0, 1.0),
        }
        self._bench = bench
        self._max_fidels = bench.max_fidels

    def __call__(self, trial: optuna.Trial) -> float:
        eval_config = dict(trial.params)
        out = self._bench(eval_config, fidels=self._max_fidels)
        trial.set_user_attr("runtime", out["runtime"])
        return out["loss"]


def optimize(n_trials: int = 400, timeout: float | None = None) -> dict:
    n_workers = 10
    if n_trials > 1000:
        n_workers = 1000
        n_trials = 10000

    bench = MFBranin()
    problem = MFBraninProblem(bench)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    simulator = AsyncOptBenchmarkSimulator(
        n_workers=n_workers,
        allow_parallel_sampling=False,
    )
    simulator.optimize(study, problem, default_runtime_func, n_trials=n_trials, timeout=timeout)

    results = AsyncOptBenchmarkSimulator.get_results_from_study(study)
    if timeout is None:
        assert len(results["cumtime"]) >= n_trials

    return results


def test_random_with_ask_and_tell() -> None:
    out = optimize()["cumtime"]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)


def test_random_with_ask_and_tell_with_max_total_eval_time() -> None:
    out = optimize(timeout=3600 * 20)["cumtime"]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)
    assert len(out) < 300  # terminated by time limit


def test_random_with_ask_and_tell_many_parallel() -> None:
    out = optimize(n_trials=10000)["cumtime"]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)


if __name__ == "__main__":
    unittest.main()
