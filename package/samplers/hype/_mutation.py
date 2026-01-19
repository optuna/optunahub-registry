from __future__ import annotations

import math
from typing import Any

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution

from ._mutations._base import BaseMutation


_NUMERICAL_DISTRIBUTIONS = (
    FloatDistribution,
    IntDistribution,
)


def perform_mutation(
    mutation: BaseMutation,
    rng: np.random.RandomState,
    search_space: BaseDistribution,
    value: float,
) -> Any:
    if isinstance(search_space, _NUMERICAL_DISTRIBUTIONS):
        lb = search_space.low
        ub = search_space.high

        mutation_value = mutation.mutation(value, rng, np.array([lb, ub]))
        return _untransform_numerical_param(
            mutation_value,
            search_space,
        )
    else:
        # For categorical variables, return None and subject to sample_independent.
        return None


def _untransform_numerical_param(
    trans_param: float, distribution: BaseDistribution, transform_log: bool = True
) -> int | float:
    d = distribution

    if isinstance(d, FloatDistribution):
        if d.log:
            param = math.exp(trans_param) if transform_log else trans_param
            if d.single():
                pass
            else:
                param = min(param, np.nextafter(d.high, d.high - 1))
        elif d.step is not None:
            param = float(
                np.clip(
                    np.round((trans_param - d.low) / d.step) * d.step + d.low,
                    d.low,
                    d.high,
                )
            )
        else:
            if d.single():
                param = trans_param
            else:
                param = min(trans_param, np.nextafter(d.high, d.high - 1))
    elif isinstance(d, IntDistribution):
        if d.log:
            if transform_log:
                param = int(np.clip(np.round(math.exp(trans_param)), d.low, d.high))
            else:
                param = int(trans_param)
        else:
            param = int(
                np.clip(
                    np.round((trans_param - d.low) / d.step) * d.step + d.low,
                    d.low,
                    d.high,
                )
            )
    else:
        assert False, "Should not reach. Unexpected distribution."

    return param
