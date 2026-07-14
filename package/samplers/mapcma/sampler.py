from __future__ import annotations

import math
import pickle
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna import logging
from optuna._imports import _LazyImport
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    import cmaes

else:
    cmaes = _LazyImport("cmaes")


_logger = logging.get_logger(__name__)

_EPS = 1e-10
# The value of system_attrs must be less than 2046 characters on RDBStorage.
_SYSTEM_ATTR_MAX_LENGTH = 2045


class _CmaEsAttrKeys(NamedTuple):
    optimizer: Callable[[], str]
    generation: Callable[[], str]
    popsize: Callable[[], str]


class MAPCMASampler(BaseSampler):
    """
    A sampler to solve  optimization using `cmaes <https://github.com/CyberAgentAILab/cmaes>`__ as the backend.

    Args:
        mean:
            Initial mean of MAPCMA.
        sigma0:
            The initial standard deviation of covariance matrix.
        n_max_resampling:
            The maximum number of resampling (default: 100).
            If the number of resampling attempts exceeds this value, the last sample will be clipped to the bounds and returned.
        seed:
            The seed of the random number generator.
        popsize:
            The population size.
        cov:
            The initial covariance matrix.
        momentum_r:
            Scaling ratio of momentum update.
        search_space:
            A dictionary of :class:`~optuna.distributions.BaseDistribution` that defines the search space.
            If this argument is :obj:`None`, the search space is estimated during the first trial.
            In this case, ``independent_sampler`` is used instead of the CatCma algorithm during the first trial.
        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.samplers.CatCmaSampler` is determined by
            :func:`~optuna.search_space.intersection_search_space()`.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.
    """

    def __init__(
        self,
        mean: dict[str, Any] | None = None,
        sigma0: float | None = None,
        seed: Optional[int] = None,
        popsize: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
        momentum_r: Optional[float] = None,
        search_space: dict[str, BaseDistribution] | None = None,
        independent_sampler: Optional[BaseSampler] = None,
    ):
        if sigma0 is not None:
            assert sigma0 > 0
        self.search_space = search_space
        self._seed = seed
        self._initial_popsize = popsize
        self._mean = mean
        self._sigma0 = sigma0
        self._cov = cov
        self._momentum_r = momentum_r
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = IntersectionSearchSpace()
        self._cma_rng = LazyRandomState(seed)

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        if self.search_space is not None:
            return self.search_space
        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution
        return search_space

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        completed_trials = self._get_trials(study)
        if len(search_space) == 1:
            _logger.warning(
                "`MapCmaSampler` only supports two or more dimensional continuous "
                "search space. `{}` is used instead of `MapCmaSampler`.".format(
                    self._independent_sampler.__class__.__name__
                )
            )
            return {}

        # MAPCMASampler only supports continuous search spaces.
        assert all(
            isinstance(distribution, FloatDistribution) for distribution in search_space.values()
        ), "`MapCmaSampler` only supports continuous or integer search spaces (categorical parameters are not supported)."

        trans = _SearchSpaceTransform(search_space)
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
            optimizer = self._init_optimizer(
                trans=trans,
            )

        solution_trials = self._get_solution_trials(completed_trials, optimizer.generation)
        if len(solution_trials) >= popsize:
            solutions: List[Tuple[np.ndarray, float]] = []
            for t in solution_trials[:popsize]:
                assert t.value is not None, "completed trials must have a value"
                solutions.append((trans.transform(t.params), t.value))
            optimizer.tell(solutions)

            # Store optimizer.
            optimizer_str = pickle.dumps(optimizer).hex()
            optimizer_attrs = self._split_optimizer_str(optimizer_str)
            for key in optimizer_attrs:
                study._storage.set_trial_system_attr(trial._trial_id, key, optimizer_attrs[key])

        seed = self._cma_rng.rng.randint(1, 2**16) + trial.number
        optimizer._rng.seed(seed)

        solution = optimizer.ask()

        generation_attr_key = self._attr_keys.generation()
        study._storage.set_trial_system_attr(
            trial._trial_id, generation_attr_key, optimizer.generation
        )
        popsize_attr_key = self._attr_keys.popsize()
        study._storage.set_trial_system_attr(trial._trial_id, popsize_attr_key, popsize)

        external_values = trans.untransform(solution)
        return external_values

    @property
    def _attr_keys(self) -> _CmaEsAttrKeys:
        attr_prefix = "cma:"

        def optimizer_key_template() -> str:
            return attr_prefix + "optimizer"

        def generation_attr_key_template() -> str:
            return attr_prefix + "generation"

        def popsize_attr_key_template() -> str:
            return attr_prefix + "popsize"

        return _CmaEsAttrKeys(
            optimizer_key_template,
            generation_attr_key_template,
            popsize_attr_key_template,
        )

    def _concat_optimizer_attrs(self, optimizer_attrs: Dict[str, str]) -> str:
        return "".join(
            optimizer_attrs["{}:{}".format(self._attr_keys.optimizer(), i)]
            for i in range(len(optimizer_attrs))
        )

    def _split_optimizer_str(self, optimizer_str: str) -> Dict[str, str]:
        optimizer_len = len(optimizer_str)
        attrs = {}
        for i in range(math.ceil(optimizer_len / _SYSTEM_ATTR_MAX_LENGTH)):
            start = i * _SYSTEM_ATTR_MAX_LENGTH
            end = min((i + 1) * _SYSTEM_ATTR_MAX_LENGTH, optimizer_len)
            attrs["{}:{}".format(self._attr_keys.optimizer(), i)] = optimizer_str[start:end]
        return attrs

    def _restore_optimizer(self, completed_trials: List[FrozenTrial]) -> Optional[cmaes.MAPCMA]:
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
    ) -> cmaes.MAPCMA:
        bounds = trans.bounds
        n_dimension = len(bounds)
        if self._mean is None:
            mean = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) / 2
        else:
            mean = trans.transform(self._mean)

        if self._sigma0 is None:
            sigma0 = np.min((bounds[:, 1] - bounds[:, 0]) / 6)
        else:
            sigma0 = self._sigma0

        assert sigma0 > 0, "sigma0 must be positive"
        sigma0 = max(sigma0, _EPS)

        seed = self._seed if self._seed is not None else self._cma_rng.rng.randint(1, 2**31 - 2)

        return cmaes.MAPCMA(
            mean=mean,
            sigma=sigma0,
            bounds=bounds,
            n_max_resampling=10 * n_dimension,
            seed=seed,
            population_size=self._initial_popsize,
            cov=self._cov,
            momentum_r=self._momentum_r,
        )

    def _get_trials(self, study: Study) -> List[FrozenTrial]:
        complete_trials = []
        for t in study._get_trials(deepcopy=False, use_cache=True):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
        return complete_trials

    def _get_solution_trials(
        self, trials: List[FrozenTrial], generation: int
    ) -> List[FrozenTrial]:
        generation_attr_key = self._attr_keys.generation()
        return [t for t in trials if generation == t.system_attrs.get(generation_attr_key, -1)]

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
