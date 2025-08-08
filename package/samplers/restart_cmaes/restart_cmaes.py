from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import math
import pickle
from typing import Any
from typing import NamedTuple
from typing import Union

import cmaes
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
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


CmaClass = Union[cmaes.CMA, cmaes.CMAwM]

_logger = logging.get_logger(__name__)

_EPS = 1e-10
# The value of system_attrs must be less than 2046 characters on RDBStorage.
_SYSTEM_ATTR_MAX_LENGTH = 2045


class _CmaEsAttrKeys(NamedTuple):
    optimizer: Callable[[int], str]
    generation: Callable[[int], str]
    popsize: Callable[[], str]
    n_restarts: Callable[[], str]
    n_restarts_with_large: str
    poptype: str
    small_n_eval: str
    large_n_eval: str


class RestartCmaEsSampler(BaseSampler):
    def __init__(
        self,
        x0: dict[str, Any] | None = None,
        sigma0: float | None = None,
        n_startup_trials: int = 1,
        independent_sampler: BaseSampler | None = None,
        warn_independent_sampling: bool = True,
        seed: int | None = None,
        *,
        restart_strategy: str | None = None,
        popsize: int | None = None,
        inc_popsize: int = 2,
        store_optimizer_state_in_storage: bool = False,
    ) -> None:
        self._x0 = x0
        self._sigma0 = sigma0
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._warn_independent_sampling = warn_independent_sampling
        self._cma_rng = LazyRandomState(seed)
        self._search_space = IntersectionSearchSpace()
        self._restart_strategy = restart_strategy
        self._initial_popsize = popsize
        self._inc_popsize = inc_popsize
        self._store_optimizer_state_in_storage = store_optimizer_state_in_storage
        if not self._store_optimizer_state_in_storage:
            self._optimizer_states_by_trial: dict[int, dict[str, Any]] = {}

        if restart_strategy not in (
            "ipop",
            "bipop",
            None,
        ):
            raise ValueError(
                "restart_strategy={} is unsupported. "
                "Please specify: 'ipop', 'bipop', or None.".format(restart_strategy)
            )

    def reseed_rng(self) -> None:
        # _cma_rng doesn't require reseeding because the relative sampling reseeds in each trial.
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial"
    ) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # `cma` cannot handle distributions that contain just a single value, so we skip
                # them. Note that the parameter values for such distributions are sampled in
                # `Trial`.
                continue

            if not isinstance(distribution, (FloatDistribution, IntDistribution)):
                # Categorical distribution is unsupported.
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
        if len(completed_trials) < self._n_startup_trials:
            return {}

        if len(search_space) == 1:
            if self._warn_independent_sampling:
                _logger.warning(
                    "`CmaEsSampler` only supports two or more dimensional continuous "
                    "search space. `{}` is used instead of `CmaEsSampler`.".format(
                        self._independent_sampler.__class__.__name__
                    )
                )
                self._warn_independent_sampling = False
            return {}

        trans = _SearchSpaceTransform(search_space, transform_0_1=True)

        if self._initial_popsize is None:
            self._initial_popsize = 4 + math.floor(3 * math.log(len(trans.bounds)))

        popsize: int = self._initial_popsize
        n_restarts: int = 0
        n_restarts_with_large: int = 0
        poptype: str = "small"
        small_n_eval: int = 0
        large_n_eval: int = 0
        if len(completed_trials) != 0:
            latest_trial = completed_trials[-1]

            if self._store_optimizer_state_in_storage:
                source = latest_trial.system_attrs
            else:
                latest_trial_id = latest_trial._trial_id
                source = self._optimizer_states_by_trial.get(latest_trial_id, {})

            attr_defaults = [
                (self._attr_keys.popsize(), self._initial_popsize),
                (self._attr_keys.n_restarts(), 0),
                (self._attr_keys.n_restarts_with_large, 0),
                (self._attr_keys.poptype, "small"),
                (self._attr_keys.small_n_eval, 0),
                (self._attr_keys.large_n_eval, 0),
            ]

            popsize, n_restarts, n_restarts_with_large, poptype, small_n_eval, large_n_eval = (
                source.get(key, default) for key, default in attr_defaults
            )

        optimizer = self._restore_optimizer(completed_trials, n_restarts)
        if optimizer is None:
            optimizer = self._init_optimizer(
                trans, study.direction, population_size=self._initial_popsize
            )

        if optimizer.dim != len(trans.bounds):
            if self._warn_independent_sampling:
                _logger.warning(
                    "`CmaEsSampler` does not support dynamic search space. "
                    "`{}` is used instead of `CmaEsSampler`.".format(
                        self._independent_sampler.__class__.__name__
                    )
                )
                self._warn_independent_sampling = False
            return {}

        # TODO(c-bata): Reduce the number of wasted trials during parallel optimization.
        # See https://github.com/optuna/optuna/pull/920#discussion_r385114002 for details.
        solution_trials = self._get_solution_trials(
            completed_trials, optimizer.generation, n_restarts
        )

        optimizer_states: dict[str, Any] = {}

        if len(solution_trials) >= popsize:
            solutions: list[tuple[np.ndarray, float]] = []
            for t in solution_trials[:popsize]:
                assert t.value is not None, "completed trials must have a value"
                if isinstance(optimizer, cmaes.CMAwM):
                    x = np.array(t.system_attrs["x_for_tell"])
                else:
                    x = trans.transform(t.params)
                y = t.value if study.direction == StudyDirection.MINIMIZE else -t.value
                solutions.append((x, y))

            optimizer.tell(solutions)

            if self._restart_strategy == "ipop" and optimizer.should_stop():
                n_restarts += 1
                popsize = popsize * self._inc_popsize
                optimizer = self._init_optimizer(
                    trans, study.direction, population_size=popsize, randomize_start_point=True
                )

            if self._restart_strategy == "bipop" and optimizer.should_stop():
                n_restarts += 1

                n_eval = popsize * optimizer.generation
                if poptype == "small":
                    small_n_eval += n_eval
                else:  # poptype == "large"
                    large_n_eval += n_eval

                if small_n_eval < large_n_eval:
                    poptype = "small"
                    popsize_multiplier = self._inc_popsize**n_restarts_with_large
                    popsize = math.floor(
                        self._initial_popsize
                        * popsize_multiplier ** (self._cma_rng.rng.uniform() ** 2)
                    )
                else:
                    poptype = "large"
                    n_restarts_with_large += 1
                    popsize = self._initial_popsize * (self._inc_popsize**n_restarts_with_large)

                optimizer = self._init_optimizer(
                    trans, study.direction, population_size=popsize, randomize_start_point=True
                )

            # Store optimizer.
            optimizer_str = pickle.dumps(optimizer).hex()
            optimizer_attrs = self._split_optimizer_str(optimizer_str, n_restarts)
            for key in optimizer_attrs:
                if self._store_optimizer_state_in_storage:
                    study._storage.set_trial_system_attr(
                        trial._trial_id, key, optimizer_attrs[key]
                    )
                else:
                    optimizer_states[key] = optimizer_attrs[key]

        # Caution: optimizer should update its seed value.
        seed = self._cma_rng.rng.randint(1, 2**16) + trial.number
        optimizer._rng.seed(seed)
        if isinstance(optimizer, cmaes.CMAwM):
            params, x_for_tell = optimizer.ask()
            if self._store_optimizer_state_in_storage:
                study._storage.set_trial_system_attr(
                    trial._trial_id, "x_for_tell", x_for_tell.tolist()
                )
            else:
                optimizer_states["x_for_tell"] = x_for_tell.tolist()
        else:
            params = optimizer.ask()

        attrs = [
            (self._attr_keys.generation(n_restarts), optimizer.generation),
            (self._attr_keys.popsize(), popsize),
            (self._attr_keys.n_restarts(), n_restarts),
            (self._attr_keys.n_restarts_with_large, n_restarts_with_large),
            (self._attr_keys.poptype, poptype),
            (self._attr_keys.small_n_eval, small_n_eval),
            (self._attr_keys.large_n_eval, large_n_eval),
        ]

        if self._store_optimizer_state_in_storage:
            for key, value in attrs:
                study._storage.set_trial_system_attr(trial._trial_id, key, value)
        else:
            optimizer_states.update(attrs)
            self._optimizer_states_by_trial[trial._trial_id] = optimizer_states

        external_values = trans.untransform(params)

        return external_values

    @property
    def _attr_keys(self) -> _CmaEsAttrKeys:
        def optimizer_key_template(restart: int) -> str:
            if self._restart_strategy is None:
                return "optimizer"
            else:
                return "{}:restart_{}:optimizer".format(self._restart_strategy, restart)

        def generation_attr_key_template(restart: int) -> str:
            if self._restart_strategy is None:
                return "generation"
            else:
                return "{}:restart_{}:generation".format(self._restart_strategy, restart)

        def popsize_attr_key_template() -> str:
            if self._restart_strategy is None:
                return "popsize"
            else:
                return "{}:popsize".format(self._restart_strategy)

        def n_restarts_attr_key_template() -> str:
            if self._restart_strategy is None:
                return "n_restarts"
            else:
                return "{}:n_restarts".format(self._restart_strategy)

        return _CmaEsAttrKeys(
            optimizer_key_template,
            generation_attr_key_template,
            popsize_attr_key_template,
            n_restarts_attr_key_template,
            "n_restarts_with_large",
            "poptype",
            "small_n_eval",
            "large_n_eval",
        )

    def _concat_optimizer_attrs(self, optimizer_attrs: dict[str, str], n_restarts: int = 0) -> str:
        return "".join(
            optimizer_attrs["{}:{}".format(self._attr_keys.optimizer(n_restarts), i)]
            for i in range(len(optimizer_attrs))
        )

    def _split_optimizer_str(self, optimizer_str: str, n_restarts: int = 0) -> dict[str, str]:
        optimizer_len = len(optimizer_str)
        attrs = {}
        for i in range(math.ceil(optimizer_len / _SYSTEM_ATTR_MAX_LENGTH)):
            start = i * _SYSTEM_ATTR_MAX_LENGTH
            end = min((i + 1) * _SYSTEM_ATTR_MAX_LENGTH, optimizer_len)
            attrs["{}:{}".format(self._attr_keys.optimizer(n_restarts), i)] = optimizer_str[
                start:end
            ]
        return attrs

    def _restore_optimizer(
        self,
        completed_trials: "list[optuna.trial.FrozenTrial]",
        n_restarts: int = 0,
    ) -> "CmaClass" | None:
        # Restore a previous CMA object.
        for trial in reversed(completed_trials):
            items = (
                trial.system_attrs.items()
                if self._store_optimizer_state_in_storage
                else self._optimizer_states_by_trial.get(trial._trial_id, {}).items()
            )
            optimizer_attrs = {
                key: value
                for key, value in items
                if key.startswith(self._attr_keys.optimizer(n_restarts))
            }
            if len(optimizer_attrs) == 0:
                continue

            optimizer_str = self._concat_optimizer_attrs(optimizer_attrs, n_restarts)
            return pickle.loads(bytes.fromhex(optimizer_str))
        return None

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
        direction: StudyDirection,
        population_size: int | None = None,
        randomize_start_point: bool = False,
    ) -> "CmaClass":
        lower_bounds = trans.bounds[:, 0]
        upper_bounds = trans.bounds[:, 1]
        n_dimension = len(trans.bounds)

        if randomize_start_point:
            mean = lower_bounds + (upper_bounds - lower_bounds) * self._cma_rng.rng.rand(
                n_dimension
            )
        elif self._x0 is None:
            mean = lower_bounds + (upper_bounds - lower_bounds) / 2
        else:
            # `self._x0` is external representations.
            mean = trans.transform(self._x0)

        if self._sigma0 is None:
            sigma0 = np.min((upper_bounds - lower_bounds) / 6)
        else:
            sigma0 = self._sigma0

        cov = None

        # Avoid ZeroDivisionError in cmaes.
        sigma0 = max(sigma0, _EPS)

        return cmaes.CMA(
            mean=mean,
            sigma=sigma0,
            cov=cov,
            bounds=trans.bounds,
            seed=self._cma_rng.rng.randint(1, 2**31 - 2),
            n_max_resampling=10 * n_dimension,
            population_size=population_size,
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
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        _logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `CmaEsSampler` "
            "(optimization performance may be degraded). "
            "`CmaEsSampler` does not support dynamic search space or `CategoricalDistribution`. "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `CmaEsSampler`, "
            "if this independent sampling is intended behavior.".format(
                param_name, trial.number, self._independent_sampler.__class__.__name__
            )
        )

    def _get_trials(self, study: "optuna.Study") -> list[FrozenTrial]:
        complete_trials = []
        for t in study._get_trials(deepcopy=False, use_cache=True):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
        return complete_trials

    def _get_solution_trials(
        self, trials: list[FrozenTrial], generation: int, n_restarts: int
    ) -> list[FrozenTrial]:
        generation_attr_key = self._attr_keys.generation(n_restarts)
        if self._store_optimizer_state_in_storage:
            return [t for t in trials if generation == t.system_attrs.get(generation_attr_key, -1)]
        else:
            return [
                t
                for t in trials
                if generation
                == self._optimizer_states_by_trial.get(t._trial_id, {}).get(
                    generation_attr_key, -1
                )
            ]

    def before_trial(self, study: optuna.Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)


def _is_compatible_search_space(
    trans: _SearchSpaceTransform, search_space: dict[str, BaseDistribution]
) -> bool:
    intersection_size = len(set(trans._search_space.keys()).intersection(search_space.keys()))
    return intersection_size == len(trans._search_space) == len(search_space)
