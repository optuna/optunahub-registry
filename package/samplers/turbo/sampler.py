from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna.samplers._base import _INDEPENDENT_SAMPLING_WARNING_TEMPLATE
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import StudyDirection
from optuna.study._multi_objective import _is_pareto_front
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    gp = _LazyImport("optuna._gp.gp")
    optim_mixed = _LazyImport("optuna._gp.optim_mixed")
    acqf_module = _LazyImport("optuna._gp.acqf")
    prior = _LazyImport("optuna._gp.prior")

import logging

from ._gp import acqf as acqf_module
from ._gp import gp as gp
from ._gp import optim_mixed
from ._gp import prior as prior
from ._gp import search_space as gp_search_space


_logger = logging.getLogger(__name__)

EPS = 1e-10


def _standardize_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clipped_values = gp.warn_and_convert_inf(values)
    means = np.mean(clipped_values, axis=0)
    stds = np.std(clipped_values, axis=0)
    standardized_values = (clipped_values - means) / np.maximum(EPS, stds)
    return standardized_values, means, stds


class TuRBOSampler(BaseSampler):
    """Sampler using Trust Region Bayesian optimization.

    Args:
        n_startup_trials:
            Number of initial trials PER TRUST REGION. Default is 2.
            As suggested in the original paper, consider setting this to 2*(number of parameters).
        n_trust_region:
            Number of trust regions. Default is 5.
        success_tolerance:
            Number of consecutive successful iterations required to expand the trust region.
            Default is 3.
        failure_tolerance:
            Number of consecutive failed iterations required to shrink the trust region.
            Default is 5. As suggested in the original paper,
            consider setting this to max(5, number of parameters).
        init_length:
            The initial size of the trust region. Defaults to 0.8.
        max_length:
            Maximum size of the trust region. If the trust region grows beyond this value,
            its size is clipped to this value. Defaults to 1.6.
        min_length:
            Minimum size of the trust region. If the trust region shrinks below this value,
            the region is regarded as fully explored and as having reached a local optimum.
            In this case, the current trust region is discarded and a new one is initialized
            with size `init_length`. Defaults to 0.5**7.
        seed:
            Random seed to initialize internal random number generator.
            Defaults to :obj:`None` (a seed is picked randomly).
        independent_sampler:
            Sampler used for initial sampling (for the first ``n_startup_trials`` trials)
            and for conditional parameters. Defaults to :obj:`None`
            (a random sampler with the same ``seed`` is used).
        deterministic_objective:
            Whether the objective function is deterministic or not.
            If :obj:`True`, the sampler will fix the noise variance of the surrogate model to
            the minimum value (slightly above 0 to ensure numerical stability).
            Defaults to :obj:`False`. Currently, all the objectives will be assume to be
            deterministic if :obj:`True`.
        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler,
            meaning that no GP model is used in the sampling.
            Note that the parameters of the first trial in a study are always sampled
            via an independent sampler, so no warning messages are emitted in this case.
    """

    def __init__(
        self,
        *,
        n_startup_trials: int = 4,
        n_trust_region: int = 5,
        success_tolerance: int = 3,
        failure_tolerance: int = 5,
        init_length: float = 0.8,
        max_length: float = 1.6,
        min_length: float = 0.5**7,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        deterministic_objective: bool = False,
        warn_independent_sampling: bool = True,
    ) -> None:
        self._rng = LazyRandomState(seed)
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._log_prior: Callable[[gp.GPRegressor], torch.Tensor] = prior.default_log_prior
        self._minimum_noise: float = prior.DEFAULT_MINIMUM_NOISE_VAR
        # We cache the kernel parameters for initial values of fitting the next time.
        # TODO(nabenabe): Make the cache lists system_attrs to make GPSampler stateless.
        self._gprs_cache_list: list[gp.GPRegressor] | None = None
        self._deterministic = deterministic_objective
        self._warn_independent_sampling = warn_independent_sampling

        # Control parameters of the acquisition function optimization.
        self._n_preliminary_samples: int = 2048
        # NOTE(nabenabe): ehvi in BoTorchSampler uses 20.
        self._n_local_search = 10
        self._tol = 1e-4

        # hyperparameters of TuRBOSampler
        self._init_length = init_length
        self._max_length = max_length
        self._min_length = min_length
        self._n_trust_region = n_trust_region
        self._success_tolerance = success_tolerance
        self._failure_tolerance = failure_tolerance

        self._trial_ids_for_trust_region: list[list[int]] = [
            [] for _ in range(self._n_trust_region)
        ]
        self._length: list[float] = [self._init_length for _ in range(self._n_trust_region)]
        self._n_consecutive_success: list[int] = [0 for _ in range(self._n_trust_region)]
        self._n_consecutive_failure: list[int] = [0 for _ in range(self._n_trust_region)]
        self._best_value_in_current_trust_region: list[float | None] = [
            None for _ in range(self._n_trust_region)
        ]
        self._cached_params_by_tr: list[dict[str, Any] | None] = [
            None for _ in range(self._n_trust_region)
        ]
        self._cached_acqf_by_tr: list[np.ndarray | None] = [
            None for _ in range(self._n_trust_region)
        ]

    def reset_trust_region(self, delete_trust_region_id: int) -> None:
        self._trial_ids_for_trust_region[delete_trust_region_id] = []
        self._length[delete_trust_region_id] = self._init_length
        self._n_consecutive_success[delete_trust_region_id] = 0
        self._n_consecutive_failure[delete_trust_region_id] = 0
        self._best_value_in_current_trust_region[delete_trust_region_id] = None

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

    def _optimize_acqf(
        self, acqf: acqf_module.BaseAcquisitionFunc, best_params: np.ndarray | None
    ) -> tuple[np.ndarray, float]:
        # Advanced users can override this method to change the optimization algorithm.
        # However, we do not make any effort to keep backward compatibility between versions.
        # Particularly, we may remove this function in future refactoring.
        assert best_params is None or len(best_params.shape) == 2
        normalized_params, acqf_val = optim_mixed.optimize_acqf_mixed(
            acqf,
            warmstart_normalized_params_array=best_params,
            n_preliminary_samples=self._n_preliminary_samples,
            n_local_search=self._n_local_search,
            tol=self._tol,
            rng=self._rng.rng,
        )
        return normalized_params, acqf_val

    def _get_best_params_for_multi_objective(
        self,
        normalized_params: np.ndarray,
        standardized_score_vals: np.ndarray,
    ) -> np.ndarray:
        pareto_params = normalized_params[
            _is_pareto_front(-standardized_score_vals, assume_unique_lexsorted=False)
        ]
        n_pareto_sols = len(pareto_params)
        # TODO(nabenabe): Verify the validity of this choice.
        size = min(self._n_local_search // 2, n_pareto_sols)
        chosen_indices = self._rng.rng.choice(n_pareto_sols, size=size, replace=False)
        return pareto_params[chosen_indices]

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        for id in range(self._n_trust_region):
            if len(self._trial_ids_for_trust_region[id]) < self._n_startup_trials:
                self._trial_ids_for_trust_region[id].append(trial._trial_id)
                return {}

        states = (TrialState.COMPLETE,)
        all_trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

        for id in range(self._n_trust_region):
            if self._cached_params_by_tr[id] is not None:
                continue

            trials = []
            for t in all_trials:
                if t._trial_id in self._trial_ids_for_trust_region[id]:
                    trials.append(t)

            if len(trials) < self._n_startup_trials:
                self._trial_ids_for_trust_region[id].append(trial._trial_id)
                return {}

            internal_search_space = gp_search_space.SearchSpace(search_space)
            normalized_params = internal_search_space.get_normalized_params(trials)

            _sign = np.array(
                [-1.0 if d == StudyDirection.MINIMIZE else 1.0 for d in study.directions]
            )
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

            gprs_list = []
            n_objectives = standardized_score_vals.shape[-1]
            is_categorical = internal_search_space.is_categorical
            for i in range(n_objectives):
                cache = self._gprs_cache_list[i] if self._gprs_cache_list is not None else None
                gprs_list.append(
                    gp.fit_kernel_params(
                        X=normalized_params,
                        Y=standardized_score_vals[:, i],
                        is_categorical=is_categorical,
                        log_prior=self._log_prior,
                        minimum_noise=self._minimum_noise,
                        gpr_cache=cache,
                        deterministic_objective=self._deterministic,
                    )
                )
            self._gprs_cache_list = gprs_list

            # note(sawa3030): TuRBOSampler currently supports single-objective optimization only.
            assert n_objectives == 1
            assert len(gprs_list) == 1
            acqf = acqf_module.LogEI(
                gpr=gprs_list[0],
                search_space=internal_search_space,
                threshold=standardized_score_vals[:, 0].max(),
            )
            best_params = normalized_params[np.argmax(standardized_score_vals), np.newaxis]

            length_scale = acqf.length_scales
            trust_region_length = (
                length_scale
                * self._length[id]
                / (np.prod(length_scale) ** (1 / len(length_scale)))
            )
            trust_region = np.empty((len(search_space), 2), dtype=float)
            trust_region[:, 0] = np.maximum(0.0, best_params - trust_region_length / 2)
            trust_region[:, 1] = np.minimum(1.0, best_params + trust_region_length / 2)
            acqf.search_space.set_trust_region(trust_region)
            normalized_param, acqf_val = self._optimize_acqf(acqf, best_params)

            self._cached_params_by_tr[id] = internal_search_space.get_unnormalized_param(
                normalized_param
            )
            self._cached_acqf_by_tr[id] = acqf_val

        trust_region_id = np.argmax(np.array(self._cached_acqf_by_tr, dtype=float))
        self._trial_ids_for_trust_region[trust_region_id].append(trial._trial_id)
        return self._cached_params_by_tr[trust_region_id]

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
        for id in range(self._n_trust_region):
            if trial._trial_id in self._trial_ids_for_trust_region[id]:
                self._cached_params_by_tr[id] = None
                self._cached_acqf_by_tr[id] = None
                if values is not None:
                    assert len(values) == 1
                    self._count_and_adjust_trust_region_length(id, values, study.direction)
                break

        self._independent_sampler.after_trial(study, trial, state, values)

    def _count_and_adjust_trust_region_length(
        self, trust_region_id: int, values: Sequence[float] | None, direction: StudyDirection
    ) -> None:
        if len(self._trial_ids_for_trust_region[trust_region_id]) >= self._n_startup_trials:
            if self._best_value_in_current_trust_region[trust_region_id] is not None:
                if values is not None:
                    best_value = self._best_value_in_current_trust_region[trust_region_id]
                    assert best_value is not None
                    if direction == StudyDirection.MINIMIZE:
                        if values[0] < best_value:
                            self._n_consecutive_success[trust_region_id] += 1
                            self._n_consecutive_failure[trust_region_id] = 0
                            self._best_value_in_current_trust_region[trust_region_id] = values[0]
                        else:
                            self._n_consecutive_success[trust_region_id] = 0
                            self._n_consecutive_failure[trust_region_id] += 1
                    else:
                        if values[0] > best_value:
                            self._n_consecutive_success[trust_region_id] += 1
                            self._n_consecutive_failure[trust_region_id] = 0
                            self._best_value_in_current_trust_region[trust_region_id] = values[0]
                        else:
                            self._n_consecutive_success[trust_region_id] = 0
                            self._n_consecutive_failure[trust_region_id] += 1
            else:
                if values is not None:
                    self._best_value_in_current_trust_region[trust_region_id] = values[0]

        if self._n_consecutive_success[trust_region_id] >= self._success_tolerance:
            self._length[trust_region_id] = min(
                self._length[trust_region_id] * 2.0, self._max_length
            )
            self._n_consecutive_success[trust_region_id] = 0
            self._n_consecutive_failure[trust_region_id] = 0
        elif self._n_consecutive_failure[trust_region_id] >= self._failure_tolerance:
            self._length[trust_region_id] = self._length[trust_region_id] / 2.0
            self._n_consecutive_success[trust_region_id] = 0
            self._n_consecutive_failure[trust_region_id] = 0
            if self._length[trust_region_id] < self._min_length:
                self.reset_trust_region(trust_region_id)
