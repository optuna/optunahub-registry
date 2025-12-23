from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import cast
from typing import TYPE_CHECKING
from typing import TypedDict

import numpy as np
import optuna
from optuna._experimental import warn_experimental_argument
from optuna._gp import optim_mixed
from optuna._gp import prior
from optuna._gp import search_space as gp_search_space
import optuna._gp.acqf
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _INDEPENDENT_SAMPLING_WARNING_TEMPLATE
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from typing_extensions import NotRequired


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study

import logging

import torch

from ._gp import acqf as acqf_module
from ._gp import gp


_logger = logging.getLogger(f"optuna.{__name__}")

EPS = 1e-10
_NOMINAL_PARAMS_KEY = "nominal_params"


class _NoiseKWArgs(TypedDict):
    uniform_input_noise_rads: NotRequired[torch.Tensor]
    normal_input_noise_stdevs: NotRequired[torch.Tensor]


def _standardize_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped_values = gp.warn_and_convert_inf(values)
    means = np.mean(clipped_values, axis=0)
    stds = np.std(clipped_values, axis=0)
    standardized_values = (clipped_values - means) / np.maximum(EPS, stds)
    return standardized_values, means, stds


class RobustGPSampler(BaseSampler):
    """Sampler using Gaussian process-based Bayesian optimization.

    The implementation mostly follows optuna.samplers.GPSampler.

    Args:
        seed:
            Random seed to initialize internal random number generator.
            Defaults to :obj:`None` (a seed is picked randomly).
        independent_sampler:
            Sampler used for initial sampling (for the first ``n_startup_trials`` trials)
            and for conditional parameters. Defaults to :obj:`None`
            (a random sampler with the same ``seed`` is used).
        n_startup_trials:
            Number of initial trials. Defaults to 10.
        deterministic_objective:
            Whether the objective function is deterministic or not.
            If :obj:`True`, the sampler will fix the noise variance of the surrogate model to
            the minimum value (slightly above 0 to ensure numerical stability).
            Defaults to :obj:`False`. Currently, all the objectives will be assume to be
            deterministic if :obj:`True`.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
            If ``constraints_func`` returns more than one value for a trial, that trial is
            considered feasible if and only if all values are equal to 0 or smaller.

            The ``constraints_func`` will be evaluated after each successful trial.
            The function won't be called when trials fail or are pruned, but this behavior is
            subject to change in future releases.
        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler,
            meaning that no GP model is used in the sampling.
            Note that the parameters of the first trial in a study are always sampled
            via an independent sampler, so no warning messages are emitted in this case.
        uniform_input_noise_rads:
            The input noise ranges for each parameter. For example, when `{"x": 0.1, "y": 0.2}`,
            the sampler assumes that +/- 0.1 is acceptable for `x` and +/- 0.2 is acceptable for
            `y`.
        normal_input_noise_stdevs:
            The input noise standard deviations for each parameter. For example, when
            `{"x": 0.1, "y": 0.2}` is given, the sampler assumes that the input noise of `x` and
            `y` follows `N(0, 0.1**2)` and `N(0, 0.2**2)`, respectively.
        const_noisy_param_names:
            The list of parameters determined externally rather than being decision variables.
            For these parameters, `suggest_float` samples random values instead of searching
            values that optimize the objective function.
        noisy_suggestion:
            If this option is enabled, suggested values will not be the nominal value but the
            nominal value plus added noise and clipping. This option is useful when a user want
            to evaluate the objective function against values with added noise rather than
            nominal values. If this option is enabled, nominal values can be retrieved using
            ``get_nominal_params``.
        nominal_ranges:
            An optional dictionary to override nominal ranges for a subset of parameters. If
            a range is specified for a parmaeter, it's nominal value is sampled from the given
            range instead of the range specified to ``suggest_float``. When ``noisy_suggestion``
            is enabled, this option is useful for avoiding clipping: if the noise range is
            +/- eps, specify [L, U] as a nominal range and specify [L-eps, U+eps] for
            ``suggest_float``.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        warn_independent_sampling: bool = True,
        uniform_input_noise_rads: dict[str, float] | None = None,
        normal_input_noise_stdevs: dict[str, float] | None = None,
        const_noisy_param_names: list[str] | None = None,
        noisy_suggestion: bool = False,
        nominal_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if uniform_input_noise_rads is None and normal_input_noise_stdevs is None:
            raise ValueError(
                "Either `uniform_input_noise_rads` or `normal_input_noise_stdevs` must be "
                "specified."
            )
        if uniform_input_noise_rads is not None and normal_input_noise_stdevs is not None:
            raise ValueError(
                "Only one of `uniform_input_noise_rads` and `normal_input_noise_stdevs` "
                "can be specified."
            )
        if const_noisy_param_names is not None:
            if uniform_input_noise_rads is not None and len(
                const_noisy_param_names & uniform_input_noise_rads.keys()
            ):
                raise ValueError(
                    "noisy parameters can be specified only in one of "
                    "`const_noisy_param_names` and `uniform_input_noise_rads`."
                )
            if normal_input_noise_stdevs is not None and len(
                const_noisy_param_names & normal_input_noise_stdevs.keys()
            ):
                raise ValueError(
                    "noisy parameters can be specified only in one of "
                    "`const_noisy_param_names` and `normal_input_noise_stdevs`."
                )

        self._uniform_input_noise_rads = uniform_input_noise_rads
        self._normal_input_noise_stdevs = normal_input_noise_stdevs
        self._const_noisy_param_names = const_noisy_param_names or []
        self._noisy_suggestion = noisy_suggestion
        self._nominal_ranges = nominal_ranges or {}
        self._rng = LazyRandomState(seed)
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        # We assume gp.GPRegressor is compatible with optuna._gp.gp.GPRegressor
        self._log_prior: Callable[[gp.GPRegressor], torch.Tensor] = cast(
            Callable[[gp.GPRegressor], torch.Tensor], prior.default_log_prior
        )
        self._minimum_noise: float = prior.DEFAULT_MINIMUM_NOISE_VAR
        # We cache the kernel parameters for initial values of fitting the next time.
        # TODO(nabenabe): Make the cache lists system_attrs to make GPSampler stateless.
        self._gprs_cache_list: list[gp.GPRegressor] | None = None
        self._constraints_gprs_cache_list: list[gp.GPRegressor] | None = None
        self._deterministic = deterministic_objective
        self._constraints_func = constraints_func
        self._warn_independent_sampling = warn_independent_sampling

        if constraints_func is not None:
            warn_experimental_argument("constraints_func")

        # Control parameters of the acquisition function optimization.
        self._n_preliminary_samples: int = 2048
        # NOTE(nabenabe): ehvi in BoTorchSampler uses 20.
        self._n_local_search = 10
        self._tol = 1e-4
        # Control parameters of value at risk.
        self._objective_confidence_level = 0.95
        self._feas_prob_confidence_level = 0.95
        self._n_input_noise_samples = 32
        self._n_qmc_samples = 128

    def _log_independent_sampling(self, trial: FrozenTrial, param_name: str) -> None:
        msg = _INDEPENDENT_SAMPLING_WARNING_TEMPLATE.format(
            param_name=param_name,
            trial_number=trial.number,
            independent_sampler_name=self._independent_sampler.__class__.__name__,
            sampler_name=self.__class__.__name__,
            fallback_reason="dynamic search space is not supported by GPSampler",
        )
        _logger.warning(msg)

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def _optimize_acqf(self, acqf: acqf_module.BaseAcquisitionFunc) -> np.ndarray:
        # Advanced users can override this method to change the optimization algorithm.
        # However, we do not make any effort to keep backward compatibility between versions.
        # Particularly, we may remove this function in future refactoring.
        normalized_params, _acqf_val = optim_mixed.optimize_acqf_mixed(
            # We assume acqf_module.BaseAcquisitionFunc is compatible with optuna._gp.acqf.BaseAcquisitionFunc
            cast(optuna._gp.acqf.BaseAcquisitionFunc, acqf),
            warmstart_normalized_params_array=None,
            n_preliminary_samples=self._n_preliminary_samples,
            n_local_search=self._n_local_search,
            tol=self._tol,
            rng=self._rng.rng,
        )
        return normalized_params

    def _get_constraints_acqf_args(
        self,
        constraint_vals: np.ndarray,
        internal_search_space: gp_search_space.SearchSpace,
        normalized_params: np.ndarray,
    ) -> tuple[list[gp.GPRegressor], list[float]]:
        # NOTE(nabenabe): Flip the sign of constraints since they are always to be minimized.
        standardized_constraint_vals, means, stds = _standardize_values(-constraint_vals)
        if (
            self._gprs_cache_list is not None
            and len(self._gprs_cache_list[0].inverse_squared_lengthscales)
            != internal_search_space.dim
        ):
            # Clear cache if the search space changes.
            self._constraints_gprs_cache_list = None

        is_categorical = internal_search_space.is_categorical
        constraints_gprs = []
        constraints_threshold_list = []
        constraints_threshold_list = (-means / np.maximum(EPS, stds)).tolist()
        for i, vals in enumerate(standardized_constraint_vals.T):
            cache = (
                self._constraints_gprs_cache_list[i]
                if self._constraints_gprs_cache_list is not None
                else None
            )
            gpr = gp.fit_kernel_params(
                X=normalized_params,
                Y=vals,
                is_categorical=is_categorical,
                log_prior=self._log_prior,
                minimum_noise=self._minimum_noise,
                gpr_cache=cache,
                deterministic_objective=self._deterministic,
            )
            constraints_gprs.append(gpr)

        self._constraints_gprs_cache_list = constraints_gprs
        return constraints_gprs, constraints_threshold_list

    def _get_internal_search_space_with_fixed_params(
        self,
        search_space: dict[str, BaseDistribution],
    ) -> gp_search_space.SearchSpace:
        search_space_with_fixed_params = search_space.copy()
        for param_name in self._const_noisy_param_names:
            search_space_with_fixed_params[param_name] = optuna.distributions.IntDistribution(0, 0)
        return gp_search_space.SearchSpace(search_space_with_fixed_params)

    def _get_value_at_risk(
        self,
        gpr: gp.GPRegressor,
        internal_search_space: gp_search_space.SearchSpace,
        search_space: dict[str, BaseDistribution],
        acqf_type: str,
        const_noisy_param_values: dict[str, float],
        constraints_gpr_list: list[gp.GPRegressor] | None = None,
        constraints_threshold_list: list[float] | None = None,
    ) -> acqf_module.ValueAtRisk | acqf_module.ConstrainedLogValueAtRisk:
        def _get_scaled_input_noise_params(
            input_noise_params: dict[str, float], noise_param_name: str
        ) -> torch.Tensor:
            if not (input_noise_params.keys() <= search_space.keys()):
                raise KeyError(
                    f"param names in {noise_param_name} must be in {list(search_space.keys())}."
                )
            discrete_dists = (
                optuna.distributions.CategoricalDistribution,
                optuna.distributions.IntDistribution,
            )
            scaled_input_noise_params = torch.zeros(len(search_space), dtype=torch.float64)
            for i, (param_name, dist) in enumerate(search_space.items()):
                if param_name not in input_noise_params:
                    continue
                err_msg = f"Cannot add input noise to discrete parameter '{param_name}'."
                if isinstance(dist, discrete_dists):
                    raise ValueError(err_msg)
                assert isinstance(dist, optuna.distributions.FloatDistribution)
                if dist.step is not None:
                    raise ValueError(err_msg)
                elif dist.log:
                    raise ValueError(
                        f"Cannot add input noise to log-scaled parameter '{param_name}'."
                    )
                input_noise_param = input_noise_params[param_name]
                scaled_input_noise_params[i] = input_noise_param / (dist.high - dist.low)
            return scaled_input_noise_params

        noise_kwargs: _NoiseKWArgs = {}
        const_noise_param_inds = [
            i
            for i, param_name in enumerate(search_space)
            if param_name in self._const_noisy_param_names
        ]

        def normalize(dist: BaseDistribution, x: float) -> float:
            assert isinstance(
                dist,
                (optuna.distributions.IntDistribution, optuna.distributions.FloatDistribution),
            )
            return (x - dist.low) / (dist.high - dist.low)

        const_noisy_param_normalized_values = [
            normalize(dist, const_noisy_param_values[param_name])
            if param_name in const_noisy_param_values
            else 0.5
            for i, (param_name, dist) in enumerate(search_space.items())
            if param_name in self._const_noisy_param_names
        ]

        if self._uniform_input_noise_rads is not None:
            scaled_input_noise_params = _get_scaled_input_noise_params(
                self._uniform_input_noise_rads, "uniform_input_noise_rads"
            )
            # FIXME(sakai): If the fixed value is not at the center of the range,
            # \pm 0.5 may not cover the domain.
            scaled_input_noise_params[const_noise_param_inds] = 0.5
            noise_kwargs["uniform_input_noise_rads"] = scaled_input_noise_params
        elif self._normal_input_noise_stdevs is not None:
            scaled_input_noise_params = _get_scaled_input_noise_params(
                self._normal_input_noise_stdevs, "normal_input_noise_stdevs"
            )
            # NOTE(nabenabe): \pm 2 sigma will cover the domain.
            # FIXME(sakai): If the fixed value is not at the center of the range,
            # \pm 2 sigma may not cover the domain.
            scaled_input_noise_params[const_noise_param_inds] = 0.25
            noise_kwargs["normal_input_noise_stdevs"] = scaled_input_noise_params
        else:
            assert False, "Should not reach here."

        search_space_with_fixed_params = self._get_internal_search_space_with_fixed_params(
            search_space
        )
        if constraints_gpr_list is None or constraints_threshold_list is None:
            return acqf_module.ValueAtRisk(
                gpr=gpr,
                search_space=search_space_with_fixed_params,
                confidence_level=self._objective_confidence_level,
                n_input_noise_samples=self._n_input_noise_samples,
                n_qmc_samples=self._n_qmc_samples,
                qmc_seed=self._rng.rng.randint(1 << 30),
                acqf_type=acqf_type,
                fixed_indices=torch.tensor(const_noise_param_inds, dtype=torch.int64),
                fixed_values=torch.tensor(
                    const_noisy_param_normalized_values, dtype=torch.float64
                ),
                **noise_kwargs,
            )
        else:
            return acqf_module.ConstrainedLogValueAtRisk(
                gpr=gpr,
                search_space=search_space_with_fixed_params,
                constraints_gpr_list=constraints_gpr_list,
                constraints_threshold_list=constraints_threshold_list,
                objective_confidence_level=self._objective_confidence_level,
                feas_prob_confidence_level=self._feas_prob_confidence_level,
                n_input_noise_samples=self._n_input_noise_samples,
                n_qmc_samples=self._n_qmc_samples,
                qmc_seed=self._rng.rng.randint(1 << 30),
                acqf_type=acqf_type,
                fixed_indices=torch.tensor(const_noise_param_inds, dtype=torch.int64),
                fixed_values=torch.tensor(
                    const_noisy_param_normalized_values, dtype=torch.float64
                ),
                **noise_kwargs,
            )

    def _verify_search_space(self, search_space: dict[str, BaseDistribution]) -> None:
        noisy_param_cands = [
            k
            for k, v in search_space.items()
            if isinstance(v, optuna.distributions.FloatDistribution)
            and v.step is None
            and not v.log
        ]
        noise_param_names: list[str]
        if self._uniform_input_noise_rads is not None:
            noise_param_names = list(self._uniform_input_noise_rads.keys())
        elif self._normal_input_noise_stdevs is not None:
            noise_param_names = list(self._normal_input_noise_stdevs.keys())
        else:
            assert "Should not reach here."

        if len(set(noisy_param_cands) & set(noise_param_names)) == 0:
            raise ValueError(
                "RobustGPSampler needs at least one noisy parameter defined in the search space. "
                f"However, no noisy parameters are found. Noisy parameter candidates are "
                f"{noisy_param_cands}, but input noise is specified for {noise_param_names}."
            )

    def _get_nominal_search_space(
        self, search_space: dict[str, BaseDistribution]
    ) -> dict[str, BaseDistribution]:
        for param_name, (low, high) in self._nominal_ranges.items():
            dist = search_space[param_name]
            assert isinstance(dist, optuna.distributions.FloatDistribution)
            assert dist.step is None
            assert not dist.log
            if low < dist.low or dist.high < high:
                raise ValueError(
                    f"The nominal range for a parameter {param_name} should be a subset of "
                    f"[{dist.low}, {dist.high}], but ([{low}, {high}]) is given."
                )
        return search_space | {
            param_name: optuna.distributions.FloatDistribution(low, high)
            for param_name, (low, high) in self._nominal_ranges.items()
        }

    def _get_gpr_list(
        self, study: Study, search_space: dict[str, BaseDistribution]
    ) -> list[gp.GPRegressor]:
        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        internal_search_space = gp_search_space.SearchSpace(search_space)
        normalized_params = internal_search_space.get_normalized_params(trials)
        signs = np.array([-1.0 if d == StudyDirection.MINIMIZE else 1.0 for d in study.directions])
        standardized_score_vals, _, _ = _standardize_values(
            signs * np.array([trial.values for trial in trials])
        )

        if (
            self._gprs_cache_list is not None
            and len(self._gprs_cache_list[0].inverse_squared_lengthscales)
            != internal_search_space.dim
        ):
            # Clear cache if the search space changes.
            self._gprs_cache_list = None
        n_objectives = standardized_score_vals.shape[-1]
        is_categorical = internal_search_space.is_categorical
        assert n_objectives == 1, "Value at risk supports only single objective."
        cache = self._gprs_cache_list[0] if self._gprs_cache_list is not None else None
        gprs_list = [
            gp.fit_kernel_params(
                X=normalized_params,
                Y=standardized_score_vals[:, 0],
                is_categorical=is_categorical,
                log_prior=self._log_prior,
                minimum_noise=self._minimum_noise,
                gpr_cache=cache,
                deterministic_objective=self._deterministic,
            )
        ]
        self._gprs_cache_list = gprs_list
        return gprs_list

    def _optimize_params(
        self,
        study: Study,
        trials: list[FrozenTrial],
        search_space: dict[str, BaseDistribution],
        const_noisy_param_values: dict[str, float],
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        self._verify_search_space(search_space)

        gprs_list = self._get_gpr_list(study, search_space)
        acqf: acqf_module.BaseAcquisitionFunc
        assert len(gprs_list) == 1
        internal_search_space = gp_search_space.SearchSpace(search_space)
        if self._constraints_func is None:
            acqf = self._get_value_at_risk(
                # TODO: Replace mean with nei once NEI is implemented.
                gprs_list[0],
                internal_search_space,
                search_space,
                acqf_type="mean",
                const_noisy_param_values=const_noisy_param_values,
            )
        else:
            constraint_vals, _ = _get_constraint_vals_and_feasibility(study, trials)
            constr_gpr_list, constr_threshold_list = self._get_constraints_acqf_args(
                constraint_vals,
                internal_search_space,
                internal_search_space.get_normalized_params(trials),
            )
            acqf = self._get_value_at_risk(
                # TODO: Replace mean with nei once NEI is implemented.
                gprs_list[0],
                internal_search_space,
                search_space,
                acqf_type="mean",
                constraints_gpr_list=constr_gpr_list,
                constraints_threshold_list=constr_threshold_list,
                const_noisy_param_values=const_noisy_param_values,
            )

        normalized_param = self._optimize_acqf(acqf)
        # The normalized values of constant noise parameters are fixed at 0.5 during search
        # regardless of their original values given as const_noisy_param_values, so
        # `internal_search_space.get_unnormalized_param` cannot decode them correctly.
        # Therefore, we overwrite those values with their original values.
        return (
            internal_search_space.get_unnormalized_param(normalized_param)
            | const_noisy_param_values
        )

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
        if len(trials) < self._n_startup_trials:
            return {}

        # Perturb constant noisy parameter uniformly
        const_noisy_param_values = {}
        for name in self._const_noisy_param_names:
            dist = search_space[name]
            assert isinstance(dist, optuna.distributions.FloatDistribution)
            const_noisy_param_values[name] = self._rng.rng.uniform(dist.low, dist.high)

        nominal_search_space = self._get_nominal_search_space(search_space)
        nominal_params = self._optimize_params(
            study, trials, nominal_search_space, const_noisy_param_values
        )

        if not self._noisy_suggestion:
            return nominal_params

        study._storage.set_trial_system_attr(trial._trial_id, _NOMINAL_PARAMS_KEY, nominal_params)

        noisy_params = dict(nominal_params)
        if self._uniform_input_noise_rads is not None:
            for param_name, rad in self._uniform_input_noise_rads.items():
                dist = search_space[param_name]
                assert isinstance(dist, optuna.distributions.FloatDistribution)
                assert dist.step is None
                assert not dist.log
                noisy_params[param_name] = min(
                    dist.high,
                    max(dist.low, nominal_params[param_name] + self._rng.rng.uniform(-rad, rad)),
                )
        elif self._normal_input_noise_stdevs is not None:
            for param_name, std in self._normal_input_noise_stdevs.items():
                dist = search_space[param_name]
                assert isinstance(dist, optuna.distributions.FloatDistribution)
                assert dist.step is None
                assert not dist.log
                noisy_params[param_name] = min(
                    dist.high,
                    max(dist.low, nominal_params[param_name] + self._rng.rng.normal(scale=std)),
                )
        else:
            assert False, "Should not reach here."

        return noisy_params

    def get_robust_trial(
        self, study: Study, const_noisy_param_nominal_values: dict[str, float] | None = None
    ) -> FrozenTrial:
        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        search_space = self._get_nominal_search_space(
            self.infer_relative_search_space(study, trials[0])
        )
        gpr = self._get_gpr_list(study, search_space)[0]
        internal_search_space = gp_search_space.SearchSpace(search_space)
        if self._nominal_ranges:
            # When nominal ranges are specified, we need to use nominal params
            # since VaR-based acquisition functions work on nominal params.
            X_train = internal_search_space.get_normalized_params(
                [
                    optuna.create_trial(
                        params=self.get_nominal_params(trial),
                        distributions=trial.distributions,
                        values=trial.values,
                        state=trial.state,
                    )
                    for trial in trials
                ]
            )
        else:
            X_train = internal_search_space.get_normalized_params(trials)

        acqf: acqf_module.BaseAcquisitionFunc
        if self._constraints_func is None:
            acqf = self._get_value_at_risk(
                gpr,
                internal_search_space,
                search_space,
                acqf_type="mean",
                const_noisy_param_values=const_noisy_param_nominal_values or {},
            )
        else:
            constraint_vals, _ = _get_constraint_vals_and_feasibility(study, trials)
            constr_gpr_list, constr_threshold_list = self._get_constraints_acqf_args(
                constraint_vals,
                internal_search_space,
                internal_search_space.get_normalized_params(trials),
            )
            acqf = self._get_value_at_risk(
                # TODO: Replace mean with nei once NEI is implemented.
                gpr,
                internal_search_space,
                search_space,
                acqf_type="mean",
                constraints_gpr_list=constr_gpr_list,
                constraints_threshold_list=constr_threshold_list,
                const_noisy_param_values=const_noisy_param_nominal_values or {},
            )

        best_idx = np.argmax(acqf.eval_acqf_no_grad(X_train)).item()
        return trials[best_idx]

    def get_robust_params(
        self, study: Study, const_noisy_param_nominal_values: dict[str, float] | None = None
    ) -> dict[str, Any]:
        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        search_space = self._get_nominal_search_space(
            self.infer_relative_search_space(study, trials[0])
        )
        return self._optimize_params(
            study, trials, search_space, const_noisy_param_nominal_values or {}
        )

    def get_nominal_params(
        self, trial: FrozenTrial, const_noisy_param_nominal_values: dict[str, float] | None = None
    ) -> dict[str, Any]:
        if _NOMINAL_PARAMS_KEY in trial.system_attrs:
            params = trial.system_attrs[_NOMINAL_PARAMS_KEY]
            if const_noisy_param_nominal_values is not None:
                params = params | const_noisy_param_nominal_values
            return params
        else:
            return trial.params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if self._warn_independent_sampling:
            states = (TrialState.COMPLETE,)
            complete_trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)
        # NOTE(sakai): Should we use nominal range here?
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._independent_sampler.after_trial(study, trial, state, values)


def _get_constraint_vals_and_feasibility(
    study: Study, trials: list[FrozenTrial]
) -> tuple[np.ndarray, np.ndarray]:
    _constraint_vals = [
        study._storage.get_trial_system_attrs(trial._trial_id).get(_CONSTRAINTS_KEY, ())
        for trial in trials
    ]
    if any(len(_constraint_vals[0]) != len(c) for c in _constraint_vals):
        raise ValueError("The number of constraints must be the same for all trials.")

    constraint_vals = np.array(_constraint_vals)
    assert len(constraint_vals.shape) == 2, "constraint_vals must be a 2d array."
    is_feasible = np.all(constraint_vals <= 0, axis=1)
    assert not isinstance(is_feasible, np.bool_), "MyPy Redefinition for NumPy v2.2.0."
    return constraint_vals, is_feasible
