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
from typing import Union

import numpy as np
import optuna
from optuna import logging
from optuna._imports import _LazyImport
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
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


class CatCmawmSampler(BaseSampler):
    """A sampler to solve mixed-variablse optimization using `cmaes <https://github.com/CyberAgentAILab/cmaes>`__ as the backend.

    Args:
        search_space:
            A dictionary of :class:`~optuna.distributions.BaseDistribution` that defines the search space.
            If this argument is :obj:`None`, the search space is estimated during the first trial.
            In this case, ``independent_sampler`` is used instead of the CatCma algorithm during the first trial.

        seed:
            A random seed for CatCmawm.

        cat_param:
            A parameter of categorical distribution.

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
        mean: Optional[np.ndarray] = None,
        cov: Optional[np.ndarray] = None,
        sigma0: Optional[float] = None,
        cat_param: Optional[np.ndarray] = None,
        independent_sampler: Optional[BaseSampler] = None,
        seed: Optional[int] = None,
        *,
        popsize: Optional[int] = None,
    ) -> None:
        self.search_space = search_space
        self._mean = mean
        self._cov = cov
        self._sigma0 = sigma0
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._cma_rng = LazyRandomState(seed)
        self._intersection_search_space = IntersectionSearchSpace()
        self._initial_popsize = popsize
        self._cat_param = cat_param

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
        float_search_space = {
            k: v for k, v in search_space.items() if isinstance(v, FloatDistribution)
        }

        float_bounds = []
        for k, v in float_search_space.items():
            float_bounds.append((v.low, v.high))

        integer_search_space = {
            k: v for k, v in search_space.items() if isinstance(v, IntDistribution)
        }

        int_values = []
        for k, v in integer_search_space.items():
            int_values.append(list(range(v.low, v.high + v.step, v.step)))

        categorical_search_space = {
            k: v for k, v in search_space.items() if isinstance(v, CategoricalDistribution)
        }

        cat_num = [len(v.choices) for v in categorical_search_space.values()]

        num_features = sum(
            [
                len(float_search_space),
                len(integer_search_space),
                len(categorical_search_space),
            ]
        )

        if self._initial_popsize is None:
            self._initial_popsize = 4 + math.floor(3 * math.log(num_features))

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
                float_bounds,
                int_values,
                cat_num,
                population_size=self._initial_popsize,
            )

        solution_trials = self._get_solution_trials(completed_trials, optimizer.generation)

        if len(solution_trials) >= popsize:
            # Prepare solutions list
            solutions: List[Tuple[cmaes.CatCMAwM.Solution, float]] = []

            for t in solution_trials[:popsize]:
                assert t.value is not None, "completed trials must have a value"
                # Convert Optuna's representation to cmaes.CatCma's internal representation.

                y = t.value if study.direction == StudyDirection.MINIMIZE else -t.value

                solution = cmaes.CatCMAwM.Solution(
                    t.user_attrs["x"],
                    t.user_attrs["z"],
                    t.user_attrs["c"],
                    t.user_attrs["_v_raw"],
                )

                solutions.append((solution, y))  # type: ignore

            optimizer.tell(solutions)

            # Store optimizer.
            optimizer_str = pickle.dumps(optimizer).hex()
            optimizer_attrs = self._split_optimizer_str(optimizer_str)
            for key in optimizer_attrs:
                study._storage.set_trial_system_attr(trial._trial_id, key, optimizer_attrs[key])

        # Caution: optimizer should update its seed value.
        seed = self._cma_rng.rng.randint(1, 2**16) + trial.number
        optimizer._rng.seed(seed)

        solution = optimizer.ask()

        generation_attr_key = self._attr_keys.generation()
        study._storage.set_trial_system_attr(
            trial._trial_id, generation_attr_key, optimizer.generation
        )
        popsize_attr_key = self._attr_keys.popsize()
        study._storage.set_trial_system_attr(trial._trial_id, popsize_attr_key, popsize)

        trial.set_user_attr("x", solution.x)
        trial.set_user_attr("z", solution.z)
        trial.set_user_attr("c", solution.c)
        trial.set_user_attr("_v_raw", solution._v_raw)

        # Convert cmaes.CatCma's internal representation to Optuna's representation.
        float_values = {k: v for k, v in zip(solution.x, float_search_space.keys())}

        integer_values = {k: v for k, v in zip(solution.z, integer_search_space.keys())}

        # cmaes.CatCma returns the categorical choice as one-hot vectors, e.g.,
        # [[True False, False], [False, True]].
        categorical_values = {
            k: categorical_search_space[k].choices[p.argmax()]
            for p, k in zip(solution.c, categorical_search_space.keys())
        }
        external_values = {**float_values, **integer_values, **categorical_values}

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
        float_bounds: List,
        int_values: List,
        cat_num: List,
        population_size: Optional[int] = None,
    ) -> "CmaClass":
        return cmaes.CatCMAwM(  # type: ignore
            x_space=float_bounds,
            z_space=int_values,
            c_space=cat_num,
            population_size=population_size,
            cat_param=self._cat_param,
            seed=self._cma_rng.rng.randint(1, 2**31 - 2),
            mean=self._mean,
            cov=self._cov,
            sigma=self._sigma0,
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
