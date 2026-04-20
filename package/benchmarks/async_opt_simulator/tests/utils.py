from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import optuna


SUBDIR_NAME = "dummy"
SIMPLE_CONFIG = {"x": 0}
ON_UBUNTU = sys.platform == "linux"
UNIT_TIME = 1e-3 if ON_UBUNTU else 5e-2


class TestProblem:
    """Wraps old-style obj_func(eval_config) -> [objectives..., runtime] to the new BaseProblem API."""

    def __init__(
        self, obj_func: Any, search_space: dict[str, optuna.distributions.BaseDistribution]
    ):
        self.search_space = search_space
        self._obj_func = obj_func

    def __call__(self, trial: optuna.Trial) -> float | list[float]:
        eval_config = dict(trial.params)
        results = self._obj_func(eval_config=eval_config)
        trial.set_user_attr("runtime", results[-1])
        objectives = [float(v) for v in results[:-1]]
        return objectives[0] if len(objectives) == 1 else objectives


class CounterSampler(optuna.samplers.BaseSampler):
    """A sampler that returns trial.number (clamped to max_count-1) for each parameter."""

    def __init__(self, sleep: float = 0.0, max_count: int | None = None):
        self._sleep = sleep
        self._max_count = max_count

    def infer_relative_search_space(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> dict:
        return {}

    def sample_relative(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial, search_space: dict
    ) -> dict:
        return {}

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        time.sleep(self._sleep)
        n = trial.number
        if self._max_count is not None:
            n = min(n, self._max_count - 1)
        return n


def get_overhead_from_study(study: optuna.Study) -> dict[str, list[float]]:
    """Extract optimizer overhead (sampling time) from all trials."""
    trials = sorted(study.trials, key=lambda t: t.number)
    return {
        "before_sample": [t.user_attrs["before_sample"] for t in trials],
        "after_sample": [t.user_attrs["after_sample"] for t in trials],
        "worker_id": [t.user_attrs["worker_id"] for t in trials],
    }


class OrderCheckConfigsForSync:
    """
    Both cases use batch size 3.

    [1] 2 worker case
    worker-0: -------------------|-----|---|---|
              1000                300   200 200
    worker-1: -------|-----      |-------  |
              400     300         400

    [2] 3 worker case.
    worker-0: -------------------|-----|
              1000                300
    worker-1: -------            |-------|
              400                 400
    worker-2: -----              |---|   |---|
              300                 200     200
    """

    def __init__(self, n_workers: int, sleeping: float = 0.0):
        loss_vals = [i for i in range(7)]
        runtimes = np.array([1000, 400, 300, 300, 400, 200, 200])
        self._results = [
            [float(loss), float(runtime)] for loss, runtime in zip(loss_vals, runtimes)
        ]
        self._ans = {
            2: np.array([400, 700, 1000, 1300, 1400, 1500, 1700]),
            3: np.array([300, 400, 1000, 1200, 1300, 1400, 1600]),
        }[n_workers]
        self._n_evals = self._ans.size
        self._sleeping = sleeping

    def __call__(self, eval_config: dict[str, int], *args: Any, **kwargs: Any) -> list[float]:
        time.sleep(self._sleeping)
        results = self._results[min(eval_config["index"], len(self._results) - 1)]
        return results


class OrderCheckConfigsForSyncWithSampleLatency:
    """
    xxx means sampling time, ooo means waiting time for the sampling for the other worker, --- means waiting time.
    Note that the first sample can be considered for Ask-and-Tell interface!
    NOTE: I supported first sample consideration for the non ask-and-tell version as well.

    [1] 2 worker case (sampling time is 200 ms)
    worker-0: xxx|-------------------|xxx|-----|---|
              200 1000                200 300   200
    worker-1: xxx|-------|-----      |xxx|-------  |
              200 400     300         200 400

    [2] 3 worker case (sampling time is 200 ms)
    worker-0: xxx|-------------------|xxx|-----|
              200 1000                200 300
    worker-1: xxx|-------            |xxx|-------|
              200 400                 200 400
    worker-2: xxx|-----              |xxx|---|
              200 300                 200 200
    """

    def __init__(self, n_workers: int):
        runtimes = np.array([1000, 400, 300, 300, 400, 200]) * UNIT_TIME
        self._ans = {
            2: np.array([600, 900, 1200, 1700, 1800, 1900]),
            3: np.array([500, 600, 1200, 1600, 1700, 1800]),
        }[n_workers] * UNIT_TIME
        loss_vals = [i for i in range(self._ans.size)]
        self._results = [
            [float(loss), float(runtime)] for loss, runtime in zip(loss_vals, runtimes)
        ]
        self._n_evals = self._ans.size

    def __call__(self, eval_config: dict[str, int], *args: Any, **kwargs: Any) -> list[float]:
        results = self._results[min(eval_config["index"], len(self._results) - 1)]
        return results


class OrderCheckConfigs:
    """
    [1] 2 worker case
    worker-0: -------------------|-|---|---|---|---|---|---|---|---|-----|
              1000              100 200 200 200 200 200 200 200 200 300
    worker-1: -----|-----|-----|-----|-----------|-----|---|-----------|-------|
              300   300   300   300   600         300   200 600         400

    [2] 4 worker case
    worker-0: -------------------|-----|-----|
              1000                300   300
    worker-1: -------|-------|-------|-------|
              400     400     400     400
    worker-2: -----|-----|-----|-----|-----|
              300   300   300   300   300
    worker-3: ---|---|---|---|---|---|---|-|
              200 200 200 200 200 200 200 100
    """

    def __init__(self, n_workers: int, sleeping: float = 0.0):
        loss_vals = [i for i in range(20)]
        runtimes = {
            2: [
                1000,
                300,
                300,
                300,
                300,
                100,
                200,
                600,
                200,
                200,
                200,
                300,
                200,
                200,
                200,
                600,
                200,
                200,
                300,
                400,
            ],
            4: [
                1000,
                400,
                300,
                200,
                200,
                300,
                400,
                200,
                200,
                300,
                200,
                400,
                300,
                200,
                300,
                200,
                300,
                400,
                300,
                100,
            ],
        }[n_workers]
        self._results = [
            [float(loss), float(runtime)] for loss, runtime in zip(loss_vals, runtimes)
        ]
        self._ans = {
            2: np.array(
                [
                    300,
                    600,
                    900,
                    1000,
                    1100,
                    1200,
                    1300,
                    1500,
                    1700,
                    1800,
                    1900,
                    2100,
                    2100,
                    2300,
                    2300,
                    2500,
                    2700,
                    2900,
                    3000,
                    3300,
                ]
            ),
            4: np.array(
                [
                    200,
                    300,
                    400,
                    400,
                    600,
                    600,
                    800,
                    800,
                    900,
                    1000,
                    1000,
                    1200,
                    1200,
                    1200,
                    1300,
                    1400,
                    1500,
                    1500,
                    1600,
                    1600,
                ]
            ),
        }[n_workers]
        self._n_evals = self._ans.size
        self._sleeping = sleeping

    def __call__(self, eval_config: dict[str, int], *args: Any, **kwargs: Any) -> list[float]:
        # Latency caused by benchmark function. We must be able to ignore it in the simulation.
        time.sleep(self._sleeping)
        results = self._results[eval_config["index"]]
        return results


class OrderCheckConfigsWithSampleLatency:
    """
    xxx means sampling time, ooo means waiting time for the sampling for the other worker, --- means waiting time.
    Note that the first sample can be considered for Ask-and-Tell interface!
    NOTE: I supported first sample consideration for the non ask-and-tell version as well.

    [1] 2 worker case (sampling time is 200 ms)
    worker-0: xxx|-----|xxx|-----------|xxx|-------|
              200 300   200 600         200 400
    worker-1: ooooxxx|---|ooxxx|---|xxx|---|
              400     200 300   200 200 200

    [2] 2 worker case for Timeout (sampling time is 200 ms)
    worker-0: xxx|-|
              200 100
    worker-1: ooooxxx|
              400

    xxx means sampling time, --- means waiting time.
    Note that the first sample can be considered for Ask-and-Tell interface!

    [1] 2 worker case (sampling time is 200 ms)
    worker-0: xxx|-----|xxx|---|xxx|-------|
              200 300   200 200 200 400
    worker-1: xxx|---|xxx|-------|xxx|---|xxx|-|
              200 200 200 400     200 200 200 100
    """

    def __init__(self, parallel_sampler: bool, timeout: bool = False):
        if parallel_sampler and not timeout:
            runtimes = np.array([300, 200, 400, 200, 400, 200, 100]) * UNIT_TIME
            self._ans = np.array([400, 500, 900, 1000, 1400, 1500, 1700]) * UNIT_TIME
        elif not parallel_sampler and timeout:
            runtimes = np.array([100] * 4) * UNIT_TIME
            self._ans = np.array([np.nan] * 4) * UNIT_TIME
        else:
            runtimes = np.array([300, 200, 600, 200, 200, 400]) * UNIT_TIME
            self._ans = np.array([500, 600, 1100, 1300, 1500, 1900]) * UNIT_TIME

        loss_vals = [i for i in range(self._ans.size)]
        self._results = [
            [float(loss), float(runtime)] for loss, runtime in zip(loss_vals, runtimes)
        ]
        self._n_evals = self._ans.size

    def __call__(self, eval_config: dict[str, int], *args: Any, **kwargs: Any) -> list[float]:
        results = self._results[eval_config["index"]]
        return results


def get_configs(index: int, unittime: float) -> np.ndarray:
    """
    [0] Slow at some points

              |0       |10       |20
              12345678901234567890123456
    Worker 1: sffffssfffff             |
    Worker 2: wsffffffsssfff           |
    Worker 3: wwsffffffwwsssssfff      |
    Worker 4: wwwsfffffwwwwwwwsssssssfff

    [1] Slow from the initialization with correct n_workers
    Usually, it does not work for most optimizers if n_workers is incorrectly set
    because opt libraries typically wait till all the workers are filled up.

              |0       |10       |20
              123456789012345678901234567890
    Worker 1: sfssfwwssssfwwwwssssssf      |
    Worker 2: wsfwsssfwwwsssssfwwwwwsssssssf

    [2] Slow from the initialization with incorrect n_workers ([2] with n_workers=4)
    Assume opt library wait till all the workers are filled up.
    `.` below stands for the waiting time due to the filling up.

              |0       |10       |20
              123456789012345678901234567
    Worker 1: sf..ssssf                 |
    Worker 2: wsf.wwwwsssssf            |
    Worker 3: wwsfwwwwwwwwwssssssf      |
    Worker 4: wwwsfwwwwwwwwwwwwwwsssssssf

    [3] No overlap

              |0       |10       |20
              1234567890123456789012345678
    Worker 1: sfffffssfffffffffffff      |
    Worker 2: wsfffffffffffffssssff      |
    Worker 3: wwsffffffffsssfffffff      |
    Worker 4: wwwsffffffffffffffffsssssfff

    The random cases were generated by:
    ```python
    size = np.random.randint(15) + 4
    print((np.random.randint(6, size=size) + 1).tolist())
    ```
    Note that I manually adapt costs of each call if their ends overlap with a start of sampling.
    It is necessary to make the test results more or less deterministic.

    [4] Random case 1

              |0       |10       |20       |30       |40       |50       |60       |70
              123456789012345678901234567890123456789012345678901234567890123456789012345
    Worker 1: sfwwssffffffwwwwwwwwwwwwwwwwwwsssssssssfffff                              |
    Worker 2: wsfffwssssfwwwwwwwwwwwssssssssfffffwwwwwwwwwwwwwwwwwwwwwwwwwssssssssssssfff
    Worker 3: wwsffffwwwwwwwwsssssssffwwwwwwwwwwwwwwwwwwwwwwwwwsssssssssssff            |
    Worker 4: wwwsffwwwwsssssfwwwwwwwwwwwwwwwwwwwwwwwssssssssssff                       |

    [5] Random case 2

              |0       |10       |20       |30       |40
              1234567890123456789012345678901234567890123
    Worker 1: sfwwssffwwssssssffffff                    |
    Worker 2: wsfffffsssfffwwwwwwwwwwwwwwwwwwsssssssssfff
    Worker 3: wwsfffffwwwwwwwwsssssssff                 |
    Worker 4: wwwsfffffwwwwwwwwwwwwwwssssssssfff        |

    [6] Random case 3

              |0       |10       |20       |30       |40       |50       |60       |70
              12345678901234567890123456789012345678901234567890123456789012345678901234
    Worker 1: sfffssfffffwwwwwwwwwwssssssssfffffwwwwwwwwwwwwwwwwwwwwwwwwwssssssssssssfff
    Worker 2: wsffffffwwsssssfffffwwwwwwwwwwwwwwwwwwssssssssssffff                     |
    Worker 3: wwsffffsssffffffwwwwwwwwwwwwwsssssssssffff                               |
    Worker 4: wwwsffffffwwwwwssssssffffwwwwwwwwwwwwwwwwwwwwwwwsssssssssssfffff         |

    [7] Random case 4

              |0       |10       |20       |30       |40       |50       |60       |70       |80       |90       |100
              12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901
    Worker 1: sffffssffffffwwwwwwwwwssssssssffffwwwwwwwwwwwwwwwwwwwwwwwwwwssssssssssssfffff                       |
    Worker 2: wsffffffwwsssssffffffwwwwwwwwwwwwwwwwwwssssssssssffffwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwssssssssssssssff
    Worker 3: wwsfffwsssffffwwwwwwwwwwwwwwwwsssssssssfffffwwwwwwwwwwwwwwwwwwwwwwwwwwwwsssssssssssssff             |
    Worker 4: wwwsfffffwwwwwwsssssssfffwwwwwwwwwwwwwwwwwwwwwwwwsssssssssssff                                      |

    [8] Random case 5

              |0       |10       |20
              123456789012345678901234567
    Worker 1: sffffwssssf               |
    Worker 2: wsffffwwwwsssssffffff     |
    Worker 3: wwsfssfffff               |
    Worker 4: wwwsffffwwwwwwwsssssssfffff
    """
    configs = [
        np.array([4, 6, 6, 5, 5, 3, 3, 3], dtype=np.float64),
        np.array([0.9] * 8, dtype=np.float64),
        np.array([0.9] * 8, dtype=np.float64),
        np.array([5, 13, 8, 16, 13, 7, 2, 3], dtype=np.float64),
        np.array([1, 3, 4, 2, 6, 1, 1, 2, 5, 5, 2, 2, 3], dtype=np.float64),
        np.array([1, 5, 5, 5, 2, 3, 6, 2, 3, 3], dtype=np.float64),
        np.array([3, 6, 4, 6, 5, 6, 5, 4, 5, 4, 4, 5, 3], dtype=np.float64),
        np.array([4, 6, 3, 5, 6, 4, 6, 3, 4, 5, 4, 2, 5, 2, 2], dtype=np.float64),
        np.array([4, 4, 1, 4, 5, 1, 6, 5], dtype=np.float64),
    ][index]
    ans = [
        np.array([5, 8, 9, 9, 12, 14, 19, 26], dtype=np.float64),
        np.array([2, 3, 5, 8, 12, 17, 23, 30], dtype=np.float64),
        np.array([2, 3, 4, 5, 9, 14, 20, 27], dtype=np.float64),
        np.array([6, 11, 15, 20, 21, 21, 21, 28], dtype=np.float64),
        np.array([2, 5, 6, 7, 11, 12, 16, 24, 35, 44, 51, 62, 75], dtype=np.float64),
        np.array([2, 7, 8, 8, 9, 13, 22, 25, 34, 43], dtype=np.float64),
        np.array([4, 7, 8, 10, 11, 16, 20, 25, 34, 42, 52, 64, 74], dtype=np.float64),
        np.array([5, 6, 8, 9, 13, 14, 21, 25, 34, 44, 53, 62, 77, 87, 101], dtype=np.float64),
        np.array([4, 5, 6, 8, 11, 11, 21, 27], dtype=np.float64),
    ][index]
    return configs * unittime, ans * unittime


def simplest_dummy_func(
    eval_config: dict[str, Any],
) -> list[float]:
    return [eval_config["x"], eval_config["x"]]


def dummy_no_fidel_func(
    eval_config: dict[str, Any],
) -> list[float]:
    return [eval_config["x"], 10]


def default_runtime_func(trial: optuna.Trial) -> float:
    return trial.user_attrs["runtime"]
