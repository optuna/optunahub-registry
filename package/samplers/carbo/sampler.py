from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import torch

from ._gp import convert_inf
from ._gp import GPRegressor
from ._gp import KernelParamsTensor
from ._optim import suggest_by_carbo


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study


EPS = 1e-10
DEFAULT_MINIMUM_NOISE_VAR = 1e-6
_WORST_ROBUST_ACQF_KEY = "worst_robust_acqf_val"
_ROBUST_PARAMS_KEY = "robust_params"


def _standardize_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped_values = convert_inf(values)
    means = np.mean(clipped_values, axis=0)
    stds = np.std(clipped_values, axis=0)
    standardized_values = (clipped_values - means) / np.maximum(EPS, stds)
    return standardized_values, means, stds


def unnormalize_params(
    params: np.ndarray, is_log: np.ndarray, lows: np.ndarray, highs: np.ndarray
) -> np.ndarray:
    assert len(params.shape) == 2
    results = params * (highs - lows) + lows
    log_lows = np.log(lows[is_log])
    log_highs = np.log(highs[is_log])
    results[:, is_log] = np.exp(params[:, is_log] * (log_highs - log_lows) + log_lows)
    return results


def normalize_params(
    params: np.ndarray, is_log: np.ndarray, lows: np.ndarray, highs: np.ndarray
) -> np.ndarray:
    assert len(params.shape) == 2
    results = (params - lows) / (highs - lows)
    log_lows = np.log(lows[is_log])
    log_highs = np.log(highs[is_log])
    results[:, is_log] = (np.log(params[:, is_log]) - log_lows) / (log_highs - log_lows)
    return results


def default_log_prior(kernel_params: KernelParamsTensor) -> torch.Tensor:
    def gamma_log_prior(x: torch.Tensor, concentration: float, rate: float) -> torch.Tensor:
        return (concentration - 1) * torch.log(x) - rate * x

    return (
        -(
            0.1 / kernel_params.inverse_squared_lengthscales
            + 0.1 * kernel_params.inverse_squared_lengthscales
        ).sum()
        + gamma_log_prior(kernel_params.kernel_scale, 2, 1)
        + gamma_log_prior(kernel_params.noise_var, 1.1, 30)
    )


def _get_dist_info_as_arrays(
    search_space: dict[str, BaseDistribution],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = len(search_space)
    is_log = np.zeros(dim, dtype=bool)
    lows = np.empty(dim, dtype=float)
    highs = np.empty(dim, dtype=float)
    for d, dist in enumerate(search_space.values()):
        assert isinstance(dist, optuna.distributions.FloatDistribution)
        is_log[d] = dist.log
        lows[d] = dist.low
        highs[d] = dist.high
    return lows, highs, is_log


def _get_params_array(
    trials: list[FrozenTrial], search_space: dict[str, BaseDistribution]
) -> np.ndarray:
    params = np.empty((len(trials), len(search_space)), dtype=float)
    for i, t in enumerate(trials):
        for d, (name, dist) in enumerate(search_space.items()):
            params[i, d] = t.params[name]
    return params


class CARBOSampler(BaseSampler):
    """Modified Constrained Adversarially Robust Bayesian Optimization (CARBO) sampler

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
        rho:
            The mix up coefficient for the acquisition function. If this value is large, the
            parameter suggestion puts more priority on constraints.
        beta:
            The coefficient for LCB and UCB. If this value is large, the parameter suggestion
            becomes more pessimistic, meaning that the search is inclined to explore more.
        local_ratio:
            The `epsilon` parameter in the CARBO algorithm that controls the size of `W(theta)`.
            This value must be in `[0, 1]`.

            .. warning::
              Use ``input_noise_rads`` instead.
        input_noise_rads:
            The input noise ranges for each parameter. For example, when `{"x": 0.1, "y": 0.2}`,
            the sampler assumes that +/- 0.1 is acceptable for `x` and +/- 0.2 is acceptable for
            `y`. This determines `W(theta)`.
        n_local_search:
            How many times the local search is performed.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        rho: float = 1e3,
        beta: float = 4.0,
        local_ratio: float | None = None,
        input_noise_rads: dict[str, float] | None = None,
        # n_local_search is a power of 2 to suppress the warning in Sobol.
        n_local_search: int = 16,
    ) -> None:
        self._rng = LazyRandomState(seed)
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._log_prior: Callable[[KernelParamsTensor], torch.Tensor] = default_log_prior
        self._minimum_noise = float(DEFAULT_MINIMUM_NOISE_VAR)
        self._kernel_params_cache: torch.Tensor | None = None
        self._constraints_kernel_params_cache_list: list[torch.Tensor] | None = None
        self._deterministic = deterministic_objective
        self._constraints_func = constraints_func
        self._beta = beta
        self._rho = rho

        if local_ratio is not None and input_noise_rads is not None:
            raise ValueError("Cannot specify both local_ratio and input_noise_rads")
        if local_ratio is None and input_noise_rads is None:
            # For backward compatibility
            local_ratio = 0.1
        if local_ratio is not None:
            assert 0 < local_ratio < 1
        self._local_ratio = local_ratio
        self._input_noise_rads = input_noise_rads

        self._n_local_search = n_local_search

    def _preproc(
        self, study: Study, trials: list[FrozenTrial], search_space: dict[str, BaseDistribution]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _sign = -1.0 if study.direction == StudyDirection.MINIMIZE else 1.0
        standardized_score_vals, _, _ = _standardize_values(
            _sign * np.array([trial.value for trial in trials])
        )
        dim = len(search_space)
        params = _get_params_array(trials, search_space)
        lows, highs, is_log = _get_dist_info_as_arrays(search_space)
        X_train = torch.from_numpy(normalize_params(params, is_log, lows, highs))
        y_train = torch.from_numpy(standardized_score_vals)
        if self._kernel_params_cache is not None and len(self._kernel_params_cache) - 2 != dim:
            self._kernel_params_cache = None
            self._constraints_kernel_params_cache_list = None

        return X_train, y_train

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if study._is_multi_objective():
            raise ValueError("CARBOSampler does not support multi-objective optimization.")
        if search_space == {}:
            return {}

        trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
        if len(trials) < self._n_startup_trials:
            return {}

        X_train, y_train = self._preproc(study, trials, search_space)
        gpr = GPRegressor(
            X_train, y_train, kernel_params=self._kernel_params_cache
        ).fit_kernel_params(self._log_prior, self._minimum_noise, self._deterministic)
        self._kernel_params_cache = gpr.kernel_params.clone()
        constraint_vals = (
            None if self._constraints_func is None else _get_constraint_vals(study, trials)
        )
        if constraint_vals is None:
            constraints_gpr_list = None
            constraints_threshold_list = None
        else:
            _cache_list = (
                self._constraints_kernel_params_cache_list
                if self._constraints_kernel_params_cache_list is not None
                else [None] * constraint_vals.shape[-1]  # type: ignore[list-item]
            )
            stded_c_vals, means, stdevs = _standardize_values(-constraint_vals)
            constraints_threshold_list = (-means / np.maximum(EPS, stdevs)).tolist()
            C_train = torch.from_numpy(stded_c_vals)
            constraints_gpr_list = [
                GPRegressor(X_train, c_train, kernel_params=cache).fit_kernel_params(
                    self._log_prior, self._minimum_noise, self._deterministic
                )
                for cache, c_train in zip(_cache_list, C_train.T)
            ]

        lows, highs, is_log = _get_dist_info_as_arrays(search_space)
        if self._local_ratio is not None:
            local_radius = np.full_like(lows, self._local_ratio / 2)
        elif self._input_noise_rads is not None:
            local_radius = np.zeros_like(lows)
            for i, (param_name, rad) in enumerate(self._input_noise_rads.items()):
                if is_log[i]:
                    raise ValueError(
                        f"Specifying input_noise_rads for log-domain parameter ({param_name}) is not supported yet."
                    )
                local_radius[i] = rad / (highs[i] - lows[i])
        else:
            assert False

        robust_params, worst_robust_params, worst_robust_acqf_val = suggest_by_carbo(
            gpr=gpr,
            constraints_gpr_list=constraints_gpr_list,
            constraints_threshold_list=constraints_threshold_list,
            rng=self._rng.rng,
            rho=self._rho,
            beta=self._beta,
            n_local_search=self._n_local_search,
            local_radius=local_radius,
        )

        robust_ext_params = {
            name: float(param_value)
            for name, param_value in zip(
                search_space, unnormalize_params(robust_params[None], is_log, lows, highs)[0]
            )
        }
        study._storage.set_trial_system_attr(
            trial._trial_id, _WORST_ROBUST_ACQF_KEY, worst_robust_acqf_val
        )
        study._storage.set_trial_system_attr(
            trial._trial_id, _ROBUST_PARAMS_KEY, robust_ext_params
        )
        return {
            name: float(param_value)
            for name, param_value in zip(
                search_space, unnormalize_params(worst_robust_params[None], is_log, lows, highs)[0]
            )
        }

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

        for distribution in search_space.values():
            if not isinstance(distribution, optuna.distributions.FloatDistribution):
                raise ValueError(f"Only FloatDistribution is supported, but got {distribution}.")
            if distribution.step is not None:
                raise ValueError("step cannot be specified in `suggest_float`.")

        return search_space

    def get_robust_params_from_trial(self, trial: FrozenTrial) -> dict[str, Any]:
        return trial.system_attrs[_ROBUST_PARAMS_KEY]

    def get_robust_params(self, study: Study) -> dict[str, Any]:
        robust_trial = self.get_robust_trial(study)
        return self.get_robust_params_from_trial(robust_trial)

    def get_robust_trial(self, study: Study) -> FrozenTrial:
        complete_trials = study._get_trials(
            deepcopy=False, states=(TrialState.COMPLETE,), use_cache=False
        )
        if len(complete_trials) == 0:
            raise ValueError("No complete trials found in the study.")

        best_idx = np.argmax(
            [t.system_attrs.get(_WORST_ROBUST_ACQF_KEY, -np.inf) for t in complete_trials]
        )
        return complete_trials[best_idx]

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


def _get_constraint_vals(study: Study, trials: list[FrozenTrial]) -> np.ndarray:
    _constraint_vals = [
        study._storage.get_trial_system_attrs(trial._trial_id).get(_CONSTRAINTS_KEY, ())
        for trial in trials
    ]
    if any(len(_constraint_vals[0]) != len(c) for c in _constraint_vals):
        raise ValueError("The number of constraints must be the same for all trials.")

    constraint_vals = np.array(_constraint_vals)
    assert len(constraint_vals.shape) == 2, "constraint_vals must be a 2d array."
    return constraint_vals
