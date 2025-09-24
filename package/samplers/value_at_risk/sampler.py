from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna._gp import optim_mixed
from optuna._gp import search_space as gp_search_space
from optuna.samplers import GPSampler
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.samplers._base import BaseSampler
    from optuna.study import Study

import logging

import torch

from ._gp import acqf as acqf_module
from ._gp import gp


_logger = logging.getLogger(f"optuna.{__name__}")

EPS = 1e-10


def _standardize_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped_values = gp.warn_and_convert_inf(values)
    means = np.mean(clipped_values, axis=0)
    stds = np.std(clipped_values, axis=0)
    standardized_values = (clipped_values - means) / np.maximum(EPS, stds)
    return standardized_values, means, stds


class RobustGPSampler(GPSampler):
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
        uniform_input_noise_ranges:
            The input noise ranges for each parameter. For example, when `{"x": 0.1, "y": 0.2}`,
            the sampler assumes that +/- 0.1 is acceptable for `x` and +/- 0.2 is acceptable for
            `y`.
        normal_input_noise_stdevs:
            The input noise standard deviations for each parameter. For example, when
            `{"x": 0.1, "y": 0.2}` is given, the sampler assumes that the input noise of `x` and
            `y` follows `N(0, 0.1**2)` and `N(0, 0.2**2)`, respectively.
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
        uniform_input_noise_ranges: dict[str, float] | None = None,
        normal_input_noise_stdevs: dict[str, float] | None = None,
    ) -> None:
        if uniform_input_noise_ranges is None and normal_input_noise_stdevs is None:
            raise ValueError(
                "Either `uniform_input_noise_ranges` or `normal_input_noise_stdevs` must be "
                "specified."
            )
        if uniform_input_noise_ranges is not None and normal_input_noise_stdevs is not None:
            raise ValueError(
                "Only one of `uniform_input_noise_ranges` and `normal_input_noise_stdevs` "
                "can be specified."
            )
        self._uniform_input_noise_ranges = uniform_input_noise_ranges
        self._normal_input_noise_stdevs = normal_input_noise_stdevs
        super().__init__(
            seed=seed,
            independent_sampler=independent_sampler,
            n_startup_trials=n_startup_trials,
            deterministic_objective=deterministic_objective,
            constraints_func=constraints_func,
            warn_independent_sampling=warn_independent_sampling,
        )
        self._gprs_cache_list: list[gp.GPRegressor] | None = None
        self._constraints_gprs_cache_list: list[gp.GPRegressor] | None = None

    def _optimize_acqf(
        self, acqf: acqf_module.BaseAcquisitionFunc, best_params: np.ndarray | None
    ) -> np.ndarray:
        # Advanced users can override this method to change the optimization algorithm.
        # However, we do not make any effort to keep backward compatibility between versions.
        # Particularly, we may remove this function in future refactoring.
        assert best_params is None or len(best_params.shape) == 2
        normalized_params, _acqf_val = optim_mixed.optimize_acqf_mixed(
            acqf,
            warmstart_normalized_params_array=best_params,
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

    def _get_value_at_risk(
        self,
        gpr: gp.GPRegressor,
        internal_search_space: gp_search_space.SearchSpace,
        search_space: dict[str, BaseDistribution],
        acqf_type: str,
    ) -> acqf_module.ValueAtRisk:
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

        if self._uniform_input_noise_ranges is not None:
            scaled_input_noise_params = _get_scaled_input_noise_params(
                self._uniform_input_noise_ranges, "uniform_input_noise_ranges"
            )
            return acqf_module.ValueAtRisk(
                gpr=gpr,
                search_space=internal_search_space,
                alpha=0.95,
                n_input_noise_samples=32,
                n_qmc_samples=128,
                qmc_seed=self._rng.rng.randint(1 << 30),
                uniform_input_noise_ranges=scaled_input_noise_params,
                acqf_type=acqf_type,
            )
        elif self._normal_input_noise_stdevs is not None:
            scaled_input_noise_params = _get_scaled_input_noise_params(
                self._normal_input_noise_stdevs, "normal_input_noise_stdevs"
            )
            return acqf_module.ValueAtRisk(
                gpr=gpr,
                search_space=internal_search_space,
                alpha=0.95,
                n_input_noise_samples=32,
                n_qmc_samples=128,
                qmc_seed=self._rng.rng.randint(1 << 30),
                normal_input_noise_stdevs=scaled_input_noise_params,
                acqf_type=acqf_type,
            )
        else:
            assert False, "Should not reach here."

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        if len(trials) < self._n_startup_trials:
            return {}

        internal_search_space = gp_search_space.SearchSpace(search_space)
        normalized_params = internal_search_space.get_normalized_params(trials)

        _sign = np.array([-1.0 if d == StudyDirection.MINIMIZE else 1.0 for d in study.directions])
        standardized_score_vals, _, _ = _standardize_values(
            _sign * np.array([trial.values for trial in trials])
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

        best_params: np.ndarray | None
        acqf: acqf_module.BaseAcquisitionFunc
        assert len(gprs_list) == 1
        if self._constraints_func is None:
            acqf = self._get_value_at_risk(
                # TODO: Replace mean with nei once NEI is implemented.
                gprs_list[0],
                internal_search_space,
                search_space,
                acqf_type="mean",
            )
            best_params = None
        else:
            assert False, "Not Implemented."
            constraint_vals, is_feasible = _get_constraint_vals_and_feasibility(study, trials)
            y_with_neginf = np.where(is_feasible, standardized_score_vals[:, 0], -np.inf)
            constr_gpr_list, constr_threshold_list = self._get_constraints_acqf_args(
                constraint_vals, internal_search_space, normalized_params
            )
            i_opt = np.argmax(y_with_neginf)
            best_feasible_y = y_with_neginf[i_opt]
            acqf = acqf_module.ConstrainedLogEI(
                gpr=gprs_list[0],
                search_space=internal_search_space,
                threshold=best_feasible_y,
                constraints_gpr_list=constr_gpr_list,
                constraints_threshold_list=constr_threshold_list,
            )
            assert normalized_params.shape[:-1] == y_with_neginf.shape
            best_params = (
                None if np.isneginf(best_feasible_y) else normalized_params[i_opt, np.newaxis]
            )

        normalized_param = self._optimize_acqf(acqf, best_params)
        return internal_search_space.get_unnormalized_param(normalized_param)

    def get_robust_trial(self, study: Study) -> FrozenTrial:
        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        search_space = self.infer_relative_search_space(study, trials[0])
        internal_search_space = gp_search_space.SearchSpace(search_space)
        X_train = internal_search_space.get_normalized_params(trials)
        _sign = np.array([-1.0 if d == StudyDirection.MINIMIZE else 1.0 for d in study.directions])
        y_train, _, _ = _standardize_values(_sign * np.array([trial.values for trial in trials]))
        is_categorical = internal_search_space.is_categorical
        gpr = gp.fit_kernel_params(
            X=X_train,
            Y=y_train.squeeze(),
            is_categorical=is_categorical,
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            gpr_cache=None,
            deterministic_objective=self._deterministic,
        )

        acqf: acqf_module.BaseAcquisitionFunc
        if self._constraints_func is None:
            acqf = self._get_value_at_risk(
                gpr, internal_search_space, search_space, acqf_type="mean"
            )
        else:
            assert False, "Not Implemented."

        best_idx = np.argmax(acqf.eval_acqf_no_grad(X_train)).item()
        return trials[best_idx]

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
