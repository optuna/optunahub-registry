from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import optuna
from optuna import logging
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)

_GENERATION_ATTR_KEY = "implicit_natural_gradient:generation"


class FastINGO:
    def __init__(
        self,
        mean: np.ndarray,
        inv_sigma: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        n_dimension = len(mean)
        if population_size is None:
            population_size = 4 + int(np.floor(3 * np.log(n_dimension)))
            population_size = 2 * (population_size // 2)

        self._learning_rate = learning_rate or 1.0 / np.sqrt(n_dimension)
        self._mean = mean
        self._inv_sigma = inv_sigma
        self._lower = lower
        self._upper = upper
        self._rng = np.random.RandomState(seed)
        self._population_size = population_size

        self._original_x = np.array([])
        self._z = np.array([])
        self._rot_z = np.array([])
        self._g = 0

    @property
    def dim(self) -> int:
        return self._mean.shape[0]

    @property
    def generation(self) -> int:
        return self._g

    @property
    def population_size(self) -> int:
        return self._population_size

    @property
    def _sigma(self) -> np.ndarray:
        return 1 / self._inv_sigma

    def ask(self) -> np.ndarray:
        dimension = self._mean.shape[0]
        z = self._rng.randn(self._population_size, dimension)
        self._z = z
        self._rot_z = z * np.sqrt(self._sigma)
        assert self._rot_z.shape == (self._population_size, dimension)
        x = self._mean + self._rot_z
        assert x.shape == (self._population_size, dimension)

        # filter
        self._original_x = x
        return np.clip(x, self._lower, self._upper)

    def tell(self, x: np.ndarray, y: np.ndarray) -> None:
        self._g += 1
        assert len(x) == len(y) == self._population_size
        dimension = self._mean.shape[0]
        y = np.array(y, dtype="f")

        # bound penalty
        x = np.array(x)
        penalty = np.mean((x - self._original_x) ** 2, axis=1)
        y = y + penalty

        y_std = np.max([np.std(y), 1e-9])
        score = (y - np.mean(y)) / y_std / self._population_size

        dx = self._rot_z.T.dot(score)
        assert dx.shape == (dimension,)
        self._mean -= self._learning_rate * dx

        z2 = self._z**2
        assert z2.shape == (self._population_size, dimension)
        ds = -z2.T.dot(score) * self._inv_sigma
        assert ds.shape == (dimension,)
        self._inv_sigma -= self._learning_rate * ds
        self._inv_sigma = np.maximum(self._inv_sigma, 1e-8)


class ImplicitNaturalGradientSampler(BaseSampler):
    """A sampler based on `Implicit Natural Gradient <https://arxiv.org/abs/1910.04301>`_
    optimization method."""

    def __init__(
        self,
        x0: Optional[Dict[str, Any]] = None,
        sigma0: Optional[float] = None,
        lr: Optional[float] = None,
        n_startup_trials: int = 1,
        independent_sampler: Optional[BaseSampler] = None,
        warn_independent_sampling: bool = True,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
    ) -> None:
        self._x0 = x0
        self._sigma0 = sigma0
        self._lr = lr
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._n_startup_trials = n_startup_trials
        self._warn_independent_sampling = warn_independent_sampling
        self._search_space = optuna.search_space.IntersectionSearchSpace()
        self._optimizer: Optional[FastINGO] = None
        self._seed = seed
        self._population_size = population_size

        # TODO(c-bata): Support multiple workers
        self._param_queue: List[Dict[str, Any]] = []

    def _get_optimizer(self) -> FastINGO:
        assert self._optimizer is not None
        return self._optimizer

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()
        if self._optimizer:
            self._optimizer._rng.seed()

    def infer_relative_search_space(
        self, study: "optuna.Study", trial: "optuna.trial.FrozenTrial"
    ) -> Dict[str, BaseDistribution]:
        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # `cma` cannot handle distributions that contain just a single value, so we skip
                # them. Note that the parameter values for such distributions are sampled in
                # `Trial`.
                continue

            if not isinstance(
                distribution,
                (
                    optuna.distributions.FloatDistribution,
                    optuna.distributions.IntDistribution,
                ),
            ):
                # Categorical distribution is unsupported.
                continue
            search_space[name] = distribution

        return search_space

    def _pop_from_param_queue(
        self, study: "optuna.Study", trial_id: int
    ) -> Optional[Dict[str, Any]]:
        if not self._param_queue:
            return None
        study._storage.set_trial_system_attr(
            trial_id, _GENERATION_ATTR_KEY, self._get_optimizer().generation
        )
        return self._param_queue.pop()

    def _check_trial_is_generation(self, trial: FrozenTrial) -> bool:
        current_gen = self._get_optimizer().generation
        trial_gen = trial.system_attrs.get(_GENERATION_ATTR_KEY, -1)
        return current_gen == trial_gen

    def sample_relative(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        self._raise_error_if_multi_objective(study)

        if len(search_space) == 0:
            return {}

        completed_trials = study.get_trials(deepcopy=True, states=[TrialState.COMPLETE])
        if len(completed_trials) < self._n_startup_trials:
            return {}

        if len(search_space) == 1:
            _logger.info(
                f"`{self.__class__.__name__}` only supports two or more dimensional"
                f" continuous search space. `{self._independent_sampler.__class__.__name__}`"
                f" is used instead of `{self.__class__.__name__}`."
            )
            self._warn_independent_sampling = False
            return {}

        trans = _SearchSpaceTransform(search_space)

        if self._optimizer is None:
            self._optimizer = self._init_optimizer(trans, population_size=self._population_size)

        if self._optimizer.dim != len(trans.bounds):
            _logger.info(
                f"`{self.__class__.__name__}` does not support dynamic search space. "
                f"`{self._independent_sampler.__class__.__name__}` is used instead of "
                f"`{self.__class__.__name__}`."
            )
            self._warn_independent_sampling = False
            return {}

        params = self._pop_from_param_queue(study, trial._trial_id)
        if params:
            return params

        solution_trials = [t for t in completed_trials if self._check_trial_is_generation(t)]
        if len(solution_trials) >= self._optimizer.population_size:
            assert (
                len(solution_trials) == self._optimizer.population_size
            ), "This sampler currently does not support PRUNED or FAIL"
            solutions_x = np.empty(shape=(self._optimizer.population_size, len(trans.bounds)))
            solutions_y = np.empty(shape=self._optimizer.population_size)
            for i, t in enumerate(solution_trials[: self._optimizer.population_size]):
                assert t.value is not None, "completed trials must have a value"
                solutions_x[i, :] = trans.transform(t.params)
                # TODO(c-bata): Check whether FastINGO assumes maximize or minimize problem
                solutions_y[i] = (
                    -t.value if study.direction == StudyDirection.MINIMIZE else t.value
                )
            self._optimizer.tell(solutions_x, solutions_y)

        # Caution: optimizer should update its seed value.
        x = self._optimizer.ask()
        assert np.isfinite(x).all(), "Detect nan or infinity"
        self._param_queue = [
            trans.untransform(x[i]) for i in range(self._optimizer.population_size)
        ]
        params = self._pop_from_param_queue(study, trial._trial_id)
        assert params is not None
        return params

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
        population_size: Optional[int] = None,
    ) -> FastINGO:
        lower_bounds = trans.bounds[:, 0]
        upper_bounds = trans.bounds[:, 1]
        n_dimension = len(trans.bounds)

        if self._x0 is None:
            mean = lower_bounds + (upper_bounds - lower_bounds) / 2
        else:
            mean = trans.transform(self._x0)

        if self._sigma0 is None:
            sigma0 = np.min((upper_bounds - lower_bounds) / 6)
        else:
            sigma0 = self._sigma0
        inv_sigma = 1 / sigma0 * np.ones(n_dimension)

        return FastINGO(
            mean=mean,
            inv_sigma=inv_sigma,
            lower=lower_bounds,
            upper=upper_bounds,
            seed=self._seed,
            population_size=population_size,
            learning_rate=self._lr,
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
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            if len(complete_trials) >= self._n_startup_trials:
                _logger.warning(
                    f"The parameter '{param_name}' in trial#{trial.number} is sampled independently"
                    f" by using `{self._independent_sampler.__class__.__name__}` instead of"
                    f" `{self.__class__.__name__}` (optimization performance may be degraded)."
                    f" `{self.__class__.__name__}` does not support dynamic search space or"
                    " `CategoricalDistribution`. You can suppress this warning by setting"
                    " `warn_independent_sampling` to `False` in the constructor of"
                    f" `{self.__class__.__name__}`, if this independent sampling is intended behavior."
                )

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def after_trial(
        self,
        study: "optuna.Study",
        trial: "optuna.trial.FrozenTrial",
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
