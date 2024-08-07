from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from optuna.distributions import CategoricalDistribution
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions


class _CustomizableParzenEstimatorParameters(NamedTuple):
    consider_prior: bool
    prior_weight: float | None
    consider_magic_clip: bool
    weights: Callable[[int], np.ndarray]
    multivariate: bool
    b_magic_exponent: float
    min_bandwidth_factor: float
    bandwidth_strategy: str
    categorical_prior_weight: float | None


def _bandwidth_hyperopt(
    mus: np.ndarray,
    low: float,
    high: float,
    step: float | None,
) -> np.ndarray:
    step_or_0 = step or 0
    sorted_indices = np.argsort(mus)
    sorted_mus_with_endpoints = np.empty(len(mus) + 2, dtype=float)
    sorted_mus_with_endpoints[0] = low - step_or_0 / 2
    sorted_mus_with_endpoints[1:-1] = mus[sorted_indices]
    sorted_mus_with_endpoints[-1] = high + step_or_0 / 2
    sorted_sigmas = np.maximum(
        sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2],
        sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1],
    )
    return sorted_sigmas[np.argsort(sorted_indices)]


def _bandwidth_optuna(
    n_observations: int,
    consider_prior: bool,
    domain_range: float,
    dim: int,
) -> np.ndarray:
    SIGMA0_MAGNITUDE = 0.2
    sigma = SIGMA0_MAGNITUDE * max(n_observations, 1) ** (-1.0 / (dim + 4)) * domain_range
    return np.full(shape=(n_observations + consider_prior,), fill_value=sigma)


def _bandwidth_scott(mus: np.ndarray) -> np.ndarray:
    std = np.std(mus, ddof=int(mus.size > 1))
    IQR = np.subtract.reduce(np.percentile(mus, [75, 25]))
    return np.full_like(mus, 1.059 * min(IQR / 1.34, std) * mus.size**-0.2)


def _clip_bandwidth(
    sigmas: np.ndarray,
    n_observations: int,
    domain_range: float,
    consider_prior: bool,
    consider_magic_clip: bool,
    b_magic_exponent: float,
    min_bandwidth_factor: float,
) -> np.ndarray:
    # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
    maxsigma = 1.0 * domain_range
    if consider_magic_clip:
        bandwidth_factor = max(
            min_bandwidth_factor, 1.0 / (1 + n_observations + consider_prior) ** b_magic_exponent
        )
        minsigma = bandwidth_factor * domain_range
    else:
        minsigma = 1e-12

    clipped_sigmas = np.asarray(np.clip(sigmas, minsigma, maxsigma))
    if consider_prior:
        clipped_sigmas[-1] = maxsigma

    return clipped_sigmas


class _CustomizableParzenEstimator(_ParzenEstimator):
    def _calculate_numerical_distributions(
        self,
        observations: np.ndarray,
        low: float,
        high: float,
        step: float | None,
        parameters: _CustomizableParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        domain_range = high - low + (step or 0)
        consider_prior = parameters.consider_prior or len(observations) == 0

        if consider_prior:
            mus = np.append(observations, [0.5 * (low + high)])
        else:
            mus = observations.copy()

        if parameters.bandwidth_strategy == "hyperopt":
            sigmas = _bandwidth_hyperopt(mus, low, high, step)
        elif parameters.bandwidth_strategy == "optuna":
            sigmas = _bandwidth_optuna(
                n_observations=len(observations),
                consider_prior=consider_prior,
                domain_range=domain_range,
                dim=len(self._search_space),
            )
        elif parameters.bandwidth_strategy == "scott":
            sigmas = _bandwidth_scott(mus)
        else:
            raise ValueError(f"Got unknown bandwidth_strategy={parameters.bandwidth_strategy}.")

        sigmas = _clip_bandwidth(
            sigmas=sigmas,
            n_observations=len(observations),
            domain_range=domain_range,
            consider_magic_clip=parameters.consider_magic_clip,
            consider_prior=consider_prior,
            b_magic_exponent=parameters.b_magic_exponent,
            min_bandwidth_factor=parameters.min_bandwidth_factor,
        )

        if step is None:
            return _BatchedTruncNormDistributions(mus, sigmas, low, high)
        else:
            return _BatchedDiscreteTruncNormDistributions(mus, sigmas, low, high, step)

    def _calculate_categorical_distributions(
        self,
        observations: np.ndarray,
        param_name: str,
        search_space: CategoricalDistribution,
        parameters: _CustomizableParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        choices = search_space.choices
        n_choices = len(choices)
        if len(observations) == 0:
            return _BatchedCategoricalDistributions(
                weights=np.full((1, n_choices), fill_value=1.0 / n_choices)
            )

        n_kernels = len(observations) + parameters.consider_prior
        observed_indices = observations.astype(int)
        if parameters.categorical_prior_weight is None:
            weights = np.full(shape=(n_kernels, n_choices), fill_value=1.0 / n_kernels)
            weights[np.arange(len(observed_indices)), observed_indices] += 1
            weights /= weights.sum(axis=1, keepdims=True)
        else:
            assert 0 <= parameters.categorical_prior_weight <= 1
            b = parameters.categorical_prior_weight
            weights = np.full(shape=(n_kernels, n_choices), fill_value=b / (n_choices - 1))
            weights[np.arange(len(observed_indices)), observed_indices] = 1 - b
            weights[-1] = 1.0 / n_choices

        return _BatchedCategoricalDistributions(weights)
