from __future__ import annotations

import copy
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
from typing import Union

import numpy as np
import optuna
from optuna import logging
from optuna._imports import _LazyImport
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    import cmaes

    CmaClass = Union[cmaes.CatCMA]  # type: ignore
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


class CatCmaSampler(BaseSampler):
    """A sampler to solve mixed-categorical optimization using `cmaes <https://github.com/CyberAgentAILab/cmaes>`__ as the backend.

    Args:
        search_space:
            A dictionary of :class:`~optuna.distributions.BaseDistribution` that defines the search space.
            If this argument is :obj:`None`, the search space is estimated during the first trial.
            In this case, ``independent_sampler`` is used instead of the CatCma algorithm during the first trial.

        x0:
            A dictionary of initial numerical parameter values that defines the initial mean for CatCma's multivariate Gaussian distribution.
            By default, the mean of ``low`` and ``high`` for each distribution is used.

        sigma0:
            Initial standard deviation of CatCma. By default, ``sigma0`` is set to
            ``min_range / 6``, where ``min_range`` denotes the minimum range of the distributions
            in the search space.

        seed:
            A random seed for CatCma.

        cat_param:
            A parameter of categorical distribution.

        margin:
            A margin (lower bound) of categorical distribution.

        min_eigenvalue:
            Lower bound of eigenvalue of multivariate Gaussian distribution.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.samplers.CatCmaSampler` is determined by
            :func:`~optuna.search_space.intersection_search_space()`.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.

        popsize:
            A population size of CatCma.

        consider_pruned_trials:
            If this is :obj:`True`, the PRUNED trials are considered for sampling.
    """

    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        x0: Optional[Dict[str, Any]] = None,
        sigma0: Optional[float] = None,
        cat_param: Optional[np.ndarray] = None,
        margin: Optional[np.ndarray] = None,
        min_eigenvalue: Optional[float] = None,
        independent_sampler: Optional[BaseSampler] = None,
        seed: Optional[int] = None,
        *,
        consider_pruned_trials: bool = False,
        popsize: Optional[int] = None,
    ) -> None:
        self.search_space = search_space
        self._x0 = x0
        self._sigma0 = sigma0
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._cma_rng = LazyRandomState(seed)
        self._intersection_search_space = IntersectionSearchSpace()
        self._consider_pruned_trials = consider_pruned_trials
        self._initial_popsize = popsize
        self._cat_param = cat_param
        self._margin = margin
        self._min_eigenvalue = min_eigenvalue

    def reseed_rng(self) -> None:
        # _cma_rng doesn't require reseeding because the relative sampling reseeds in each trial.
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial"
    ) -> Dict[str, BaseDistribution]:
        if self.search_space is not None:
            return self.search_space
        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if distribution.single():
                # Single value objects are not sampled with the `sample_relative` method,
                # but with the `sample_independent` method.
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        completed_trials = self._get_trials(study)

        if len(search_space) == 1:
            _logger.warning(
                "`CatCmaSampler` only supports two or more dimensional continuous "
                "search space. `{}` is used instead of `CatCmaSampler`.".format(
                    self._independent_sampler.__class__.__name__
                )
            )
            return {}

        # cmaes.CatCma handles numerical and categorical parameters separately.
        # In the following, we split the search space into numerical and categorical parameters.
        # Then, for numerical parameters, we normalize them to [0, 1].
        # For categorical parameters, we convert them to the number of choices, e.g.,
        # c1 = ['a', 'b', 'c'], c2 = ['d', 'e'] -> cat_num = [3, 2].
        numerical_search_space = {
            k: v for k, v in search_space.items() if not isinstance(v, CategoricalDistribution)
        }
        trans = _SearchSpaceTransform(
            numerical_search_space, transform_step=True, transform_0_1=True
        )
        categorical_search_space = {
            k: v for k, v in search_space.items() if isinstance(v, CategoricalDistribution)
        }
        cat_num = np.asarray([len(v.choices) for v in categorical_search_space.values()])

        if self._initial_popsize is None:
            self._initial_popsize = 4 + math.floor(3 * math.log(len(trans.bounds)))

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
            optimizer = self._init_optimizer(trans, cat_num, population_size=self._initial_popsize)

        solution_trials = self._get_solution_trials(completed_trials, optimizer.generation)

        if len(solution_trials) >= popsize:
            # Calculate the number of categorical variables and maximum number of choices
            num_categorical_vars = len(categorical_search_space)
            max_num_choices = max(
                len(space.choices) for space in categorical_search_space.values()
            )

            # Prepare solutions list
            solutions: List[Tuple[Tuple[np.ndarray, np.ndarray], float]] = []

            for t in solution_trials[:popsize]:
                assert t.value is not None, "completed trials must have a value"
                # Convert Optuna's representation to cmaes.CatCma's internal representation.

                # Convert numerical parameters
                x = trans.transform({k: t.params[k] for k in numerical_search_space.keys()})

                # Convert categorial values to one-hot vectors.
                # Example:
                #   choices = ['a', 'b', 'c']
                #   value = 'b'
                #   one_hot_vec = [False, True, False]
                c = np.zeros((num_categorical_vars, max_num_choices))

                for idx, k in enumerate(categorical_search_space.keys()):
                    choices = categorical_search_space[k].choices
                    v = t.params.get(k)
                    if v is not None:
                        index = choices.index(v)
                        c[idx, index] = 1

                y = t.value if study.direction == StudyDirection.MINIMIZE else -t.value
                solutions.append(((x, c), y))  # type: ignore

            optimizer.tell(solutions)

            # Store optimizer.
            optimizer_str = pickle.dumps(optimizer).hex()
            optimizer_attrs = self._split_optimizer_str(optimizer_str)
            for key in optimizer_attrs:
                study._storage.set_trial_system_attr(trial._trial_id, key, optimizer_attrs[key])

        # Caution: optimizer should update its seed value.
        seed = self._cma_rng.rng.randint(1, 2**16) + trial.number
        optimizer._rng.seed(seed)
        params, cat_params = optimizer.ask()

        generation_attr_key = self._attr_keys.generation()
        study._storage.set_trial_system_attr(
            trial._trial_id, generation_attr_key, optimizer.generation
        )
        popsize_attr_key = self._attr_keys.popsize()
        study._storage.set_trial_system_attr(trial._trial_id, popsize_attr_key, popsize)

        # Convert cmaes.CatCma's internal representation to Optuna's representation.
        numerical_values = trans.untransform(params)
        # cmaes.CatCma returns the categorical choice as one-hot vectors, e.g.,
        # [[True False, False], [False, True]].
        categorical_values = {
            k: categorical_search_space[k].choices[p.argmax()]
            for p, k in zip(cat_params, categorical_search_space.keys())
        }
        external_values = {**numerical_values, **categorical_values}

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

    def _restore_optimizer(
        self,
        completed_trials: "List[optuna.trial.FrozenTrial]",
    ) -> Optional["CmaClass"]:
        # Restore a previous CatCma object.
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
        cat_num: np.ndarray,
        population_size: Optional[int] = None,
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
        return cmaes.CatCMA(  # type: ignore
            mean=mean,
            sigma=sigma0,
            cat_num=cat_num,
            bounds=trans.bounds,
            n_max_resampling=10 * n_dimension,
            seed=self._cma_rng.rng.randint(1, 2**31 - 2),
            population_size=population_size,
            cov=cov,
            cat_param=self._cat_param,
            margin=self._margin,
            min_eigenvalue=self._min_eigenvalue,
        )

    def sample_independent(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _get_trials(self, study: "optuna.Study") -> List[FrozenTrial]:
        complete_trials = []
        for t in study._get_trials(deepcopy=False, use_cache=True):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
            elif (
                t.state == TrialState.PRUNED
                and len(t.intermediate_values) > 0
                and self._consider_pruned_trials
            ):
                _, value = max(t.intermediate_values.items())
                if value is None:
                    continue
                # We rewrite the value of the trial `t` for sampling, so we need a deepcopy.
                copied_t = copy.deepcopy(t)
                copied_t.value = value
                complete_trials.append(copied_t)
        return complete_trials

    def _get_solution_trials(
        self, trials: List[FrozenTrial], generation: int
    ) -> List[FrozenTrial]:
        generation_attr_key = self._attr_keys.generation()
        return [t for t in trials if generation == t.system_attrs.get(generation_attr_key, -1)]

    def before_trial(self, study: optuna.Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
