from __future__ import annotations

from numpy import ndarray
from numpy.random import RandomState

from ._base import BaseMutation


class GaussMutation(BaseMutation):
    def __init__(self, sigma_factor: float = 0.1) -> None:
        self._sigma_factor = sigma_factor

    def mutation(self, value: float, rng: RandomState, search_space_bonds: ndarray) -> float:
        delta = search_space_bonds[1] - search_space_bonds[0]
        sigma = self._sigma_factor * delta
        child_param = rng.normal(value, sigma)

        return child_param
