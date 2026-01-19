from __future__ import annotations

import abc

import numpy as np


class BaseMutation(abc.ABC):
    def __str__(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def mutation(
        self,
        value: float,
        rng: np.random.RandomState,
        search_space_bounds: np.ndarray,
    ) -> float:
        """Mutate the given value.

        Args:
            value:
                The value to mutate.
            rng:
                An instance of ``numpy.random.RandomState``.
            search_space_bounds:
                A ``numpy.ndarray`` with dimensions ``len_search_space x 2`` representing
                numerical distribution bounds constructed from transformed search space.

        Returns:
            A mutated value.
        """

        raise NotImplementedError
