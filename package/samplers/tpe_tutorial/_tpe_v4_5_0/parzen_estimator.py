from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution

from .probability_distributions import _BatchedDistributions


EPS = 1e-12


class _ParzenEstimatorParameters(NamedTuple): ...


class _ParzenEstimator:
    def __init__(
        self,
        observations: dict[str, np.ndarray],
        search_space: dict[str, BaseDistribution],
        parameters: _ParzenEstimatorParameters,
        predetermined_weights: np.ndarray | None = None,
    ) -> None:
        assert False

    def sample(self, rng: np.random.RandomState, size: int) -> dict[str, np.ndarray]:
        sampled = self._mixture_distribution.sample(rng, size)
        return self._untransform(sampled)

    def log_pdf(self, samples_dict: dict[str, np.ndarray]) -> np.ndarray:
        transformed_samples = self._transform(samples_dict)
        return self._mixture_distribution.log_pdf(transformed_samples)

    @staticmethod
    def _call_weights_func(weights_func: Callable[[int], np.ndarray], n: int) -> np.ndarray:
        w = np.array(weights_func(n))[:n]
        if np.any(w < 0):
            raise ValueError(
                f"The `weights` function is not allowed to return negative values {w}. "
                + f"The argument of the `weights` function is {n}."
            )
        if len(w) > 0 and np.sum(w) <= 0:
            raise ValueError(
                f"The `weight` function is not allowed to return all-zero values {w}."
                + f" The argument of the `weights` function is {n}."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError(
                "The `weights`function is not allowed to return infinite or NaN values "
                + f"{w}. The argument of the `weights` function is {n}."
            )

        # TODO(HideakiImamura) Raise `ValueError` if the weight function returns an ndarray of
        # unexpected size.
        return w

    @staticmethod
    def _is_log(dist: BaseDistribution) -> bool:
        return isinstance(dist, (FloatDistribution, IntDistribution)) and dist.log

    def _transform(self, samples_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.array(
            [
                (
                    np.log(samples_dict[param])
                    if self._is_log(self._search_space[param])
                    else samples_dict[param]
                )
                for param in self._search_space
            ]
        ).T

    def _untransform(self, samples_array: np.ndarray) -> dict[str, np.ndarray]:
        res = {
            param: (
                np.exp(samples_array[:, i])
                if self._is_log(self._search_space[param])
                else samples_array[:, i]
            )
            for i, param in enumerate(self._search_space)
        }
        # TODO(contramundum53): Remove this line after fixing log-Int hack.
        return {
            param: (
                np.clip(
                    dist.low + np.round((res[param] - dist.low) / dist.step) * dist.step,
                    dist.low,
                    dist.high,
                )
                if isinstance(dist, IntDistribution)
                else res[param]
            )
            for (param, dist) in self._search_space.items()
        }

    def _calculate_distributions(
        self,
        transformed_observations: np.ndarray,
        param_name: str,
        search_space: BaseDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        if isinstance(search_space, CategoricalDistribution):
            return self._calculate_categorical_distributions(
                transformed_observations, param_name, search_space, parameters
            )
        else:
            assert isinstance(search_space, (FloatDistribution, IntDistribution))
            if search_space.log:
                low = np.log(search_space.low)
                high = np.log(search_space.high)
            else:
                low = search_space.low
                high = search_space.high
            step = search_space.step

            # TODO(contramundum53): This is a hack and should be fixed.
            if step is not None and search_space.log:
                low = np.log(search_space.low - step / 2)
                high = np.log(search_space.high + step / 2)
                step = None

            return self._calculate_numerical_distributions(
                transformed_observations, low, high, step, parameters
            )

    def _calculate_categorical_distributions(
        self,
        observations: np.ndarray,
        param_name: str,
        search_space: CategoricalDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        assert False

    def _calculate_numerical_distributions(
        self,
        observations: np.ndarray,
        low: float,
        high: float,
        step: float | None,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        assert False
