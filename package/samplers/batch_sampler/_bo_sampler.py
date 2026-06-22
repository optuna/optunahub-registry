from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING
import warnings

from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


@dataclass
class DimSpec:
    """Describes one dimension of the search space."""

    name: str
    type: str  # "float" | "int"
    low: float
    high: float
    log: bool = False
    step: float | None = None


SuggestFn = Callable[
    [list[list[float]], list[float], list[DimSpec], int],
    list[dict[str, Any]],
]


class BatchSampler(BaseSampler):
    """Optuna sampler that coordinates parallel workers into batches.

    When ``study.optimize(n_jobs=q)`` drives concurrent ``sample_relative``
    calls, default samplers suggest independently — each worker sees only an
    incomplete view of the current batch, producing redundant suggestions and
    wasting the throughput gained from parallelism.

    ``BatchSampler`` addresses this with a shared lock: the first worker to
    find an empty cache calls ``suggest_fn`` once to obtain ``q`` suggestions,
    then hands them out one at a time.  Other workers wait on the lock and
    pop from the already-filled cache without triggering another call.

    Falls back to independent random sampling during startup (fewer than
    ``n_startup_trials`` complete trials) and whenever ``suggest_fn`` raises.

    Parameters
    ----------
    search_space:
        Dimensions of the optimisation problem.
    suggest_fn:
        Callable with signature::

            suggest_fn(
                X: list[list[float]],    # completed trial params, shape (n, d)
                y: list[float],          # raw objective values
                search_space: list[DimSpec],
                q: int,                  # number of suggestions to return
            ) -> list[dict[str, Any]]   # exactly q parameter dicts

        ``y`` values follow the study's direction convention: lower is better
        for ``direction="minimize"``, higher is better for
        ``direction="maximize"``.  ``suggest_fn`` is responsible for handling
        the direction if the underlying acquisition function requires it.
    n_startup_trials:
        Number of random trials to run before ``suggest_fn`` is called.
    q:
        Batch size — number of suggestions requested per ``suggest_fn`` call.
        Should match ``n_jobs`` in ``study.optimize``.
    seed:
        Seed for the fallback random sampler.
    """

    def __init__(
        self,
        search_space: list[DimSpec],
        suggest_fn: SuggestFn,
        n_startup_trials: int = 8,
        q: int = 4,
        seed: int | None = None,
    ) -> None:
        self._search_space = search_space
        self._suggest_fn = suggest_fn
        self._n_startup_trials = n_startup_trials
        self._q = q
        self._independent_sampler = RandomSampler(seed=seed)
        self._pending: deque[dict[str, Any]] = deque()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # BaseSampler interface
    # ------------------------------------------------------------------

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        result: dict[str, BaseDistribution] = {}
        for dim in self._search_space:
            if dim.type == "float":
                result[dim.name] = FloatDistribution(dim.low, dim.high, log=dim.log, step=dim.step)
            elif dim.type == "int":
                result[dim.name] = IntDistribution(
                    int(dim.low),
                    int(dim.high),
                    log=dim.log,
                    step=int(dim.step) if dim.step is not None else 1,
                )
        return result

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        with self._lock:
            if self._pending:
                return self._pending.popleft()

            complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if len(complete_trials) < self._n_startup_trials:
                return {}

            param_names = [dim.name for dim in self._search_space]
            usable = [
                t
                for t in complete_trials
                if all(n in t.params for n in param_names) and t.value is not None
            ]
            if len(usable) < self._n_startup_trials:
                return {}

            X = [[float(t.params[n]) for n in param_names] for t in usable]
            y = [float(t.value) for t in usable]  # type: ignore[arg-type]

            try:
                candidates = self._suggest_fn(X, y, self._search_space, self._q)
            except Exception as exc:
                warnings.warn(
                    f"BatchSampler: suggest_fn raised {exc!r}, falling back to random.",
                    stacklevel=2,
                )
                return {}

            for params in candidates:
                self._pending.append(params)

            return self._pending.popleft() if self._pending else {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
