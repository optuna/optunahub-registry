from __future__ import annotations

from typing import Any

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution

from ._mutations._base import BaseMutation


def perform_mutation(
    mutation: BaseMutation,
    rng: np.random.RandomState,
    search_space: BaseDistribution,
    value: float,
) -> Any:
    if isinstance(search_space, FloatDistribution):
        lb = search_space.low
        ub = search_space.high

        return mutation.mutation(value, rng, np.array([lb, ub]))
    else:
        raise NotImplementedError
