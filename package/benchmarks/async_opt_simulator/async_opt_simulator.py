from __future__ import annotations

import time
from typing import TYPE_CHECKING
import warnings

import numpy as np
import optuna
from optuna.trial import TrialState


if TYPE_CHECKING:
    from typing import Final
    from typing import Protocol

    from optunahub.benchmarks import BaseProblem

    class RuntimeFunc(Protocol):
        def __call__(self, trial: optuna.Trial) -> float:
            raise NotImplementedError


_NEGLIGIBLE_SEC: Final[float] = 1e-12
# NOTE(nabenabe): The prefix `optuna.` enables us to use the optuna logger externally.
_logger = optuna.logging.get_logger(f"optuna.{__name__}")


class AsyncOptBenchmarkSimulator:
    def __init__(self, n_workers: int, allow_parallel_sampling: bool = False) -> None:
        """a simulator class for async optimization using zero-cost benchmark without waiting.

        Args:
            n_workers (int):
                The number of simulated workers. In other words, how many parallel workers to simulate.
            allow_parallel_sampling (bool):
                Whether sampling can happen in parallel.
        """
        self._n_workers = n_workers
        self._allow_parallel_sampling = allow_parallel_sampling
        if allow_parallel_sampling:
            warnings.warn(
                "allow_parallel_sampling=True uses an imprecise simulation. "
                "Results may not accurately reflect the behavior of expensive samplers."
            )
        self._worker_indices = np.arange(n_workers)

        # --- Data associated with a simulation ---
        self._timenow: float
        self._cumtimes: np.ndarray
        self._pending_results: list[tuple[int, list[float]] | None]
        self._after_sample_times: list[float]
        self._init_simulation_data()

    def _init_simulation_data(self) -> None:
        self._timenow = 0.0
        self._cumtimes = np.zeros(self._n_workers, dtype=float)
        self._pending_results = [None] * self._n_workers
        self._after_sample_times = []

    def _proc_obj_func(
        self, trial: optuna.Trial, problem: BaseProblem, runtime_func: RuntimeFunc, worker_id: int
    ) -> None:
        output = problem(trial)
        trial.set_user_attr("worker_id", worker_id)
        self._cumtimes[worker_id] += runtime_func(trial)
        trial.set_user_attr("cumtime", self._cumtimes[worker_id].item())
        self._pending_results[worker_id] = (
            trial.number,
            [output] if isinstance(output, float) else list(output),
        )

    def _ask_with_timer(
        self, study: optuna.Study, problem: BaseProblem, worker_id: int
    ) -> optuna.Trial:
        start = time.time()
        trial = study.ask(problem.search_space)
        sampling_time = time.time() - start
        is_first_sample = bool(self._cumtimes[worker_id] < _NEGLIGIBLE_SEC)
        if self._allow_parallel_sampling:
            before_sample = self._cumtimes[worker_id]
            self._cumtimes[worker_id] = self._cumtimes[worker_id] + sampling_time
        else:
            before_sample = max(self._timenow, self._cumtimes[worker_id])
            self._timenow = before_sample + sampling_time
            self._cumtimes[worker_id] = self._timenow

        trial.set_user_attr("before_sample", before_sample)
        trial.set_user_attr("after_sample", self._cumtimes[worker_id])
        self._after_sample_times.append(self._cumtimes[worker_id])
        if (
            self._allow_parallel_sampling
            and is_first_sample
            and self._cumtimes[worker_id] > _NEGLIGIBLE_SEC
            and self._cumtimes[worker_id]
            != np.min(self._cumtimes[self._cumtimes > _NEGLIGIBLE_SEC])
        ):
            raise TimeoutError(
                "The initialization of the optimizer must be cheaper than one objective evuation.\n"
                "In principle, n_workers is too large for the objective to simulate correctly.\n"
                "Please set allow_parallel_sampling=False or a smaller n_workers, or use a cheaper initialization.\n"
            )
        return trial

    def _tell_pending_result(self, study: optuna.Study, worker_id: int) -> None:
        free_worker_idxs = np.array([worker_id], dtype=int)
        if not self._allow_parallel_sampling:
            # NOTE: The cutoff uses timenow (= _after_sample_times[-1]) rather than
            # max(timenow, cumtimes[worker_id]). Under the assumption of continuous
            # runtime distribution), where cumtimes are almost surely distinct.
            before_eval = self._after_sample_times[-1]
            free_worker_idxs = np.union1d(
                self._worker_indices[self._cumtimes <= before_eval], free_worker_idxs
            )

        for _worker_id in free_worker_idxs.astype(int).tolist():
            result = self._pending_results[_worker_id]
            if result is None:
                continue

            trial_number, values = result
            study.tell(trial_number, values)
            _logger.info(f"Trial {trial_number} ({worker_id=}) finished with values: {values}.")
            self._pending_results[_worker_id] = None

    @staticmethod
    def get_results_from_study(
        study: optuna.Study, states: TrialState | None = None
    ) -> dict[str, list]:
        """Extract results sorted by cumtime."""
        valid_states = (TrialState.COMPLETE, TrialState.PRUNED)
        states = states or valid_states
        if any(s not in valid_states for s in states):
            raise ValueError(f"{states=} cannot contain states other than COMPLETE and PRUNED.")

        trials = [
            t for t in study.get_trials(deepcopy=False, states=states) if "cumtime" in t.user_attrs
        ]
        sorted_trials = sorted(trials, key=lambda t: t.user_attrs["cumtime"])
        return {
            "cumtime": [t.user_attrs["cumtime"] for t in sorted_trials],
            "values": [list(t.values) for t in sorted_trials],
            "worker_id": [t.user_attrs["worker_id"] for t in sorted_trials],
        }

    def optimize(
        self,
        study: optuna.Study,
        problem: BaseProblem,
        runtime_func: RuntimeFunc,
        *,
        n_trials: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Start the async optimization using zero-cost benchmark without any sleep.

        Args:
            n_trials (int):
                How many trials we would like to collect.
            timeout (float):
                The maximum total evaluation time for the optimization (in simulated time but not the actual runtime).
        """
        worker_id = 0
        n_trials = n_trials or 2**20  # Sufficiently large number to finish optimizing.
        timeout = timeout or float("inf")
        for i in range(n_trials + self._n_workers - 1):
            trial = self._ask_with_timer(study, problem, worker_id=worker_id)
            self._proc_obj_func(
                trial=trial, problem=problem, runtime_func=runtime_func, worker_id=worker_id
            )
            worker_id = np.argmin(self._cumtimes).item()
            if i + 1 >= self._n_workers:
                # This `if` is needed for the compatibility with the other modes.
                # It ensures that all workers filled out first.
                self._tell_pending_result(study=study, worker_id=worker_id)
            if self._cumtimes[worker_id] > timeout:  # exceed time limit
                break

        self._init_simulation_data()
