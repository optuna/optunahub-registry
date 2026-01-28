from __future__ import annotations

from numpy import ndarray
from numpy.random import RandomState

from ._base import BaseMutation


class UniformMutation(BaseMutation):
    def mutation(self, value: float, rng: RandomState, search_space_bounds: ndarray) -> float:
        delta = search_space_bounds[1] - search_space_bounds[0]
        return delta * rng.rand() + search_space_bounds[0]
