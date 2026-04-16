from __future__ import annotations

import unittest

import numpy as np
import optuna
import pytest

from .. import AsyncOptBenchmarkSimulator
from .utils import CounterSampler
from .utils import default_runtime_func
from .utils import OrderCheckConfigs
from .utils import OrderCheckConfigsWithSampleLatency
from .utils import TestProblem
from .utils import UNIT_TIME


N_EVALS = 20
LATENCY = "latency"


def optimize_parallel(mode: str, n_workers: int, parallel_sampler: bool = False, timeout: bool = False):
    latency = mode == LATENCY
    target = OrderCheckConfigsWithSampleLatency(parallel_sampler, timeout) if latency else OrderCheckConfigs(n_workers)
    n_evals = target._n_evals

    if latency:
        sampler = CounterSampler(sleep=UNIT_TIME * 200, max_count=n_evals)
    else:
        sampler = CounterSampler(max_count=n_evals)

    search_space = {"index": optuna.distributions.IntDistribution(0, max(n_evals - 1, 0))}
    problem = TestProblem(obj_func=target, search_space=search_space)
    study = optuna.create_study(sampler=sampler)
    simulator = AsyncOptBenchmarkSimulator(
        n_workers=n_workers,
        allow_parallel_sampling=parallel_sampler,
    )
    simulator.optimize(study, problem, default_runtime_func, n_trials=n_evals)

    out = np.array(AsyncOptBenchmarkSimulator.get_results_from_study(study)["cumtime"])[:n_evals]
    diffs = np.abs(out - np.maximum.accumulate(out))
    assert np.allclose(diffs, 0.0)
    diffs = np.abs(out - target._ans)
    buffer = UNIT_TIME * 100 if latency else 1
    assert np.all(diffs < buffer)  # 1 is just a buffer.


@pytest.mark.parametrize("mode", ("normal", LATENCY))
@pytest.mark.parametrize("parallel_sampler", (True, False))
def test_optimize_parallel(mode: str, parallel_sampler: bool):
    if mode == LATENCY:
        optimize_parallel(mode=mode, n_workers=2, parallel_sampler=parallel_sampler)
    elif not parallel_sampler:
        optimize_parallel(mode=mode, n_workers=2)
        optimize_parallel(mode=mode, n_workers=4)
    else:
        pass


if __name__ == "__main__":
    unittest.main()
