from __future__ import annotations

import math
import pickle
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import Sequence

from cmaes.safe_cma import SafeCMA
import numpy as np
import optuna
from optuna import logging
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)

_EPS = 1e-10
# The value of system_attrs must be less than 2046 characters on RDBStorage.
_SYSTEM_ATTR_MAX_LENGTH = 2045


class _CmaEsAttrKeys(NamedTuple):
    optimizer: Callable[[], str]
    generation: Callable[[], str]
    popsize: Callable[[], str]
    best_eval: Callable[[], str]
    unsafe_eval_counts: Callable[[], str]


class SafeCMASampler(BaseSampler):
    """A sampler using SafeCMA from `cmaes <https://github.com/CyberAgentAILab/cmaes>`__.

    SafeCMA is a variant of CMA-ES that provides additional safety mechanisms.

    Args:
        safe_seeds:
            Initial safe seed points for SafeCMA. Must be provided along with ``seeds_evals``
            and ``seeds_safe_evals``. Can be a list of lists or 2D array.
            Shape should be ``(n_seeds, n_dimensions)``.

        seeds_evals:
            Evaluation values (objective function values) for the safe seeds.
            Must be provided when ``safe_seeds`` is specified. Can be a list or array of floats.
            Shape should be ``(n_seeds,)``.

        seeds_safe_evals:
            Safe evaluation values (safety function values) for the safe seeds.
            Must be provided when ``safe_seeds`` is specified. Can be a list or array of floats.
            Shape should be ``(n_seeds, 1)`` or ``(n_seeds,)``.

        safety_threshold:
            Safety threshold values. Values from ``safe_function`` should be <= this threshold.
            Can be a list or array of floats.

        safe_function:
            Safety function that takes a sequence of parameter values and returns a safety value.
            The function receives parameter values in the same order as the search space.
            The sampler will automatically calculate ``safe_value`` for each trial using this function.

        sigma0:
            Initial standard deviation of SafeCMA. By default, ``sigma0`` is set to
            ``min_range / 6``, where ``min_range`` denotes the minimum range of the distributions
            in the search space.

        seed:
            A random seed for SafeCMA.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.samplers.SafeCMASampler` is determined by
            :func:`~optuna.search_space.intersection_search_space()`.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.

        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.

            Note that the parameters of the first trial in a study are always sampled
            via an independent sampler, so no warning messages are emitted in this case.

        popsize:
            A population size of SafeCMA.

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).

        cov:
            A covariance matrix (optional).
    """

    def __init__(
        self,
        safe_seeds: Sequence[Sequence[float]],
        seeds_evals: Sequence[float],
        seeds_safe_evals: Sequence[float] | Sequence[Sequence[float]],
        safety_threshold: Sequence[float],
        safe_function: Callable[[Sequence[float | int]], float],
        sigma0: float | None = None,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        warn_independent_sampling: bool = True,
        *,
        popsize: int | None = None,
        n_max_resampling: int = 100,
        cov: Sequence[Sequence[float]] | None = None,
    ) -> None:
        assert (
            len(safe_seeds)
            == len(seeds_evals)
            == len(seeds_safe_evals)
            == len(safety_threshold)
            != 0
        ), "The length of safe_seeds, seeds_evals, seeds_safe_evals, and safety_threshold must be the same"
        self._safe_seeds = np.array(safe_seeds, dtype=np.float64)
        self._seeds_evals = np.array(seeds_evals, dtype=np.float64)
        seeds_safe_evals_arr = np.array(seeds_safe_evals, dtype=np.float64)
        if seeds_safe_evals_arr.ndim == 1:
            seeds_safe_evals_arr = seeds_safe_evals_arr.reshape(-1, 1)
        self._seeds_safe_evals = seeds_safe_evals_arr
        self._safety_threshold = np.array(safety_threshold, dtype=np.float64)
        self._safe_function = safe_function
        self._sigma0 = sigma0
        self._cov = np.array(cov, dtype=np.float64) if cov is not None else None
        self._n_max_resampling = n_max_resampling
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._warn_independent_sampling = warn_independent_sampling
        self._seed = seed
        self._cma_rng = LazyRandomState(seed)
        self._search_space = IntersectionSearchSpace()
        self._initial_popsize = popsize
        self._attr_prefix = "safecma:"

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial"
    ) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                continue

            if not isinstance(distribution, (FloatDistribution, IntDistribution)):
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        completed_trials = self._get_trials(study)

        assert all(
            isinstance(distribution, FloatDistribution) for distribution in search_space.values()
        ), "`SafeCMASampler` only supports continuous search spaces."

        trans = _SearchSpaceTransform(search_space, transform_step=True, transform_0_1=True)

        if self._initial_popsize is None:
            self._initial_popsize = 4 + math.floor(3 * math.log(len(search_space)))

        popsize: int = self._initial_popsize
        if len(completed_trials) != 0:
            latest_trial = completed_trials[-1]
            popsize_attr_key = self._attr_keys.popsize()
            if popsize_attr_key in latest_trial.system_attrs:
                popsize = latest_trial.system_attrs[popsize_attr_key]
            else:
                popsize = self._initial_popsize

        optimizer = self._restore_optimizer(completed_trials)
        if optimizer is None:
            optimizer = self._init_optimizer(trans)

        if optimizer.dim != len(trans.bounds):
            if self._warn_independent_sampling:
                _logger.warning(
                    "`SafeCMASampler` does not support dynamic search space. "
                    "`{}` is used instead of `SafeCMASampler`.".format(
                        self._independent_sampler.__class__.__name__
                    )
                )
                self._warn_independent_sampling = False
            return {}

        solution_trials = self._get_solution_trials(completed_trials, optimizer.generation)

        if len(solution_trials) >= popsize:
            solutions: list[tuple[np.ndarray, float, np.ndarray]] = []
            best_eval = (
                float("inf") if study.direction == StudyDirection.MINIMIZE else float("-inf")
            )
            unsafe_eval_counts = 0

            for t in solution_trials[:popsize]:
                assert t.value is not None, "completed trials must have a value"
                param_names = list(trans._search_space.keys())
                x = np.array([t.params[name] for name in param_names], dtype=np.float64)
                y = t.value if study.direction == StudyDirection.MINIMIZE else -t.value

                param_list = [t.params[name] for name in param_names]
                safe_value = np.array([self._safe_function(param_list)])

                if study.direction == StudyDirection.MINIMIZE:
                    best_eval = min(best_eval, t.value)
                else:
                    best_eval = max(best_eval, t.value)

                unsafe_eval_counts += int(np.any(safe_value > self._safety_threshold))

                solutions.append((x, y, safe_value))

            optimizer.tell(solutions)

            optimizer_str = pickle.dumps(optimizer).hex()
            optimizer_attrs = self._split_optimizer_str(optimizer_str)
            for key in optimizer_attrs:
                study._storage.set_trial_system_attr(trial._trial_id, key, optimizer_attrs[key])

            best_eval_attr_key = self._attr_keys.best_eval()
            study._storage.set_trial_system_attr(trial._trial_id, best_eval_attr_key, best_eval)
            unsafe_eval_counts_attr_key = self._attr_keys.unsafe_eval_counts()
            study._storage.set_trial_system_attr(
                trial._trial_id, unsafe_eval_counts_attr_key, unsafe_eval_counts
            )

        seed = self._cma_rng.rng.randint(1, 2**16) + trial.number
        optimizer._rng.seed(seed)
        params = optimizer.ask()

        generation_attr_key = self._attr_keys.generation()
        study._storage.set_trial_system_attr(
            trial._trial_id, generation_attr_key, optimizer.generation
        )
        popsize_attr_key = self._attr_keys.popsize()
        study._storage.set_trial_system_attr(trial._trial_id, popsize_attr_key, popsize)

        param_names = list(trans._search_space.keys())
        params_dict = {name: float(params[i]) for i, name in enumerate(param_names)}

        return params_dict

    @property
    def _attr_keys(self) -> _CmaEsAttrKeys:
        attr_prefix = self._attr_prefix

        def optimizer_key_template() -> str:
            return attr_prefix + "optimizer"

        def generation_attr_key_template() -> str:
            return attr_prefix + "generation"

        def popsize_attr_key_template() -> str:
            return attr_prefix + "popsize"

        def best_eval_attr_key_template() -> str:
            return attr_prefix + "best_eval"

        def unsafe_eval_counts_attr_key_template() -> str:
            return attr_prefix + "unsafe_eval_counts"

        return _CmaEsAttrKeys(
            optimizer_key_template,
            generation_attr_key_template,
            popsize_attr_key_template,
            best_eval_attr_key_template,
            unsafe_eval_counts_attr_key_template,
        )

    def _concat_optimizer_attrs(self, optimizer_attrs: dict[str, str]) -> str:
        return "".join(
            optimizer_attrs["{}:{}".format(self._attr_keys.optimizer(), i)]
            for i in range(len(optimizer_attrs))
        )

    def _split_optimizer_str(self, optimizer_str: str) -> dict[str, str]:
        optimizer_len = len(optimizer_str)
        attrs = {}
        for i in range(math.ceil(optimizer_len / _SYSTEM_ATTR_MAX_LENGTH)):
            start = i * _SYSTEM_ATTR_MAX_LENGTH
            end = min((i + 1) * _SYSTEM_ATTR_MAX_LENGTH, optimizer_len)
            attrs["{}:{}".format(self._attr_keys.optimizer(), i)] = optimizer_str[start:end]
        return attrs

    def _restore_optimizer(
        self,
        completed_trials: "list[optuna.trial.FrozenTrial]",
    ) -> "SafeCMA" | None:
        for trial in reversed(completed_trials):
            optimizer_attrs = {
                key: value
                for key, value in trial.system_attrs.items()
                if key.startswith(self._attr_keys.optimizer())
            }
            if len(optimizer_attrs) == 0:
                continue

            optimizer_str = self._concat_optimizer_attrs(optimizer_attrs)
            return pickle.loads(bytes.fromhex(optimizer_str))
        return None

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
    ) -> "SafeCMA":
        bounds_list = []
        for dist in trans._search_space.values():
            if hasattr(dist, "low") and hasattr(dist, "high"):
                bounds_list.append([dist.low, dist.high])
            else:
                bounds_list.append([0.0, 1.0])

        bounds = np.array(bounds_list, dtype=np.float64)
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        n_dimension = len(bounds)

        safe_seeds = np.array(self._safe_seeds, dtype=np.float64)
        if safe_seeds.ndim == 1:
            safe_seeds = safe_seeds.reshape(1, -1)

        seeds_evals = np.array(self._seeds_evals, dtype=np.float64)
        seeds_safe_evals = np.array(self._seeds_safe_evals, dtype=np.float64)
        if seeds_safe_evals.ndim == 1:
            seeds_safe_evals = seeds_safe_evals.reshape(-1, 1)

        if self._sigma0 is None:
            sigma0 = np.min((upper_bounds - lower_bounds) / 6)
        else:
            sigma0 = self._sigma0

        sigma0 = max(sigma0, _EPS)

        n_max_resampling = (
            self._n_max_resampling if self._n_max_resampling != 100 else 10 * n_dimension
        )

        return SafeCMA(
            safe_seeds=safe_seeds,
            seeds_evals=seeds_evals,
            seeds_safe_evals=seeds_safe_evals,
            safety_threshold=self._safety_threshold,
            sigma=sigma0,
            bounds=bounds,
            seed=self._seed,
            n_max_resampling=n_max_resampling,
            population_size=self._initial_popsize,
            cov=self._cov,
        )

    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)

        if self._warn_independent_sampling:
            complete_trials = self._get_trials(study)
            if len(complete_trials) > 0:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        from optuna import logging

        _logger = logging.get_logger(__name__)
        _logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `{}` (i.e., {}). "
            "Note that this warning will be shown only once.".format(
                param_name,
                trial.number,
                self._independent_sampler.__class__.__name__,
                self.__class__.__name__,
                (
                    "dynamic search space and `CategoricalDistribution` are not supported "
                    "by `SafeCMASampler`"
                ),
            )
        )

    def _get_trials(self, study: "optuna.Study") -> list[FrozenTrial]:
        complete_trials = []
        for t in study._get_trials(deepcopy=False, use_cache=True):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
        return complete_trials

    def _get_solution_trials(
        self, trials: list[FrozenTrial], generation: int
    ) -> list[FrozenTrial]:
        generation_attr_key = self._attr_keys.generation()
        return [t for t in trials if generation == t.system_attrs.get(generation_attr_key, -1)]

    def before_trial(self, study: "optuna.Study", trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
