from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
from optuna import Study
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers.nsgaii._crossover import _is_contained
from optuna.samplers.nsgaii._crossover import _try_crossover
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.trial import FrozenTrial


_NUMERICAL_DISTRIBUTIONS = (
    FloatDistribution,
    IntDistribution,
)


class MOEAdChildGenerationStrategy:
    """Generate a child parameter from the given parent population by MOEA/D algorithm.
    Args:
        study:
            Target study object.
        search_space:
            A dictionary containing the parameter names and parameter's distributions.
        parent_population:
            A list of trials that are selected as parent population.
    Returns:
        A dictionary containing the parameter names and parameter's values.
    """

    def __init__(
        self,
        *,
        mutation_prob: float | None = None,
        crossover: BaseCrossover,
        crossover_prob: float,
        swapping_prob: float,
        rng: LazyRandomState,
    ) -> None:
        if not (mutation_prob is None or 0.0 <= mutation_prob <= 1.0):
            raise ValueError(
                "`mutation_prob` must be None or a float value within the range [0.0, 1.0]."
            )

        if not (0.0 <= crossover_prob <= 1.0):
            raise ValueError("`crossover_prob` must be a float value within the range [0.0, 1.0].")

        if not (0.0 <= swapping_prob <= 1.0):
            raise ValueError("`swapping_prob` must be a float value within the range [0.0, 1.0].")

        if not isinstance(crossover, BaseCrossover):
            raise ValueError(
                f"'{crossover}' is not a valid crossover."
                " For valid crossovers see"
                " https://optuna.readthedocs.io/en/stable/reference/samplers.html."
            )

        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._swapping_prob = swapping_prob
        self._crossover = crossover
        self._rng = rng

    def __call__(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        parent_population: list[FrozenTrial],
        neighbors: dict[int, list[int]],
    ) -> dict[str, Any]:
        """Generate a child parameter from the given parent population by NSGA-II algorithm.
        Args:
            study:
                Target study object.
            search_space:
                A dictionary containing the parameter names and parameter's distributions.
            parent_population:
                A list of trials that are selected as parent population.
        Returns:
            A dictionary containing the parameter names and parameter's values.
        """
        # Check: Does it work properly when parallelized?
        subproblem_id = len(study.trials) % len(neighbors)
        subproblem_parent_population = [parent_population[i] for i in neighbors[subproblem_id]]

        # We choose a child based on the specified crossover method.
        if self._rng.rng.rand() < self._crossover_prob:
            child_params = self._perform_crossover(
                self._crossover,
                study,
                subproblem_parent_population,
                search_space,
                self._rng.rng,
                self._swapping_prob,
            )
        else:
            parent_population_size = len(parent_population)
            parent_params = parent_population[self._rng.rng.choice(parent_population_size)].params
            child_params = {name: parent_params[name] for name in search_space.keys()}

        n_params = len(child_params)
        if self._mutation_prob is None:
            mutation_prob = 1.0 / max(1.0, n_params)
        else:
            mutation_prob = self._mutation_prob

        params = {}
        for param_name in child_params.keys():
            if self._rng.rng.rand() >= mutation_prob:
                params[param_name] = child_params[param_name]
        return params

    def _perform_crossover(
        self,
        crossover: BaseCrossover,
        study: Study,
        parent_population: Sequence[FrozenTrial],
        search_space: Dict[str, BaseDistribution],
        rng: np.random.RandomState,
        swapping_prob: float,
    ) -> Dict[str, Any]:
        numerical_search_space: Dict[str, BaseDistribution] = {}
        categorical_search_space: Dict[str, BaseDistribution] = {}
        for key, value in search_space.items():
            if isinstance(value, _NUMERICAL_DISTRIBUTIONS):
                numerical_search_space[key] = value
            else:
                categorical_search_space[key] = value

        numerical_transform: Optional[_SearchSpaceTransform] = None
        if len(numerical_search_space) != 0:
            numerical_transform = _SearchSpaceTransform(numerical_search_space)

        while True:  # Repeat while parameters lie outside search space boundaries.
            parents = self._select_parents(crossover, parent_population, rng)
            child_params = _try_crossover(
                parents,
                crossover,
                study,
                rng,
                swapping_prob,
                categorical_search_space,
                numerical_search_space,
                numerical_transform,
            )

            if _is_contained(child_params, search_space):
                break

        return child_params

    def _select_parents(
        self,
        crossover: BaseCrossover,
        parent_population: Sequence[FrozenTrial],
        rng: np.random.RandomState,
    ) -> List[FrozenTrial]:
        parents: List[FrozenTrial] = rng.choice(
            np.array(parent_population), crossover.n_parents, replace=False
        ).tolist()
        return parents
