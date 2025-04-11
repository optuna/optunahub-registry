from __future__ import annotations

from numpy import ndarray
from numpy.random import RandomState

from ._base import BaseMutation


class PolynomialMutation(BaseMutation):
    def __init__(
        self,
        eta: float = 20.0,
    ) -> None:
        self._eta = eta

    def mutation(self, value: float, rng: RandomState, search_space_bonds: ndarray) -> float:
        u = rng.rand()
        lb = search_space_bonds[0]
        ub = search_space_bonds[1]

        if u <= 0.5:
            delta_l = (2.0 * u) ** (1.0 / (self._eta + 1.0)) - 1.0
            child_param = value + delta_l * (value - lb)
        else:
            delta_r = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (self._eta + 1.0))
            child_param = value + delta_r * (ub - value)

        return child_param
