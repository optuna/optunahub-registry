from __future__ import annotations
from collections import defaultdict
import hashlib
from typing import Any, Dict
from optuna import Study
import optuna
from optuna.samplers import BaseSampler, RandomSampler
from optuna.distributions import BaseDistribution
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import FrozenTrial


_GENERATION_KEY = "moead:generation"
_POPULATION_CACHE_KEY_PREFIX = "moead:population"


class MOEAdSampler(BaseSampler):
    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        seed: int | None = None,
        population_size: int = 50,
        num_weight_vectors: int = 100,
        num_decomposition: int = 3,
        scalar_aggregation_function: str = "weighted_sum",
    ) -> None:
        """Multi-objective sampler using the MOEA/D algorithm.

        MOEA/D stands for "Multi-Objective Evolutionary Algorithm based on Decomposition.

        For more information about MOEA/D, please refer to the following paper:

        - `MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition<https://doi.org/10.1109/TEVC.2007.892759>`__

        Args:
            search_space:
                A dictionary containing the parameter names and their distributions.
            seed:
                Seed for random number generator.
            scalar_aggregation_function:
                The scalar aggregation function to use. The default is "weighted_sum". Other options are "tchebycheff" and "PBI".
        """
        self._search_space = search_space
        self._seed = seed
        self._intersection_search_space = IntersectionSearchSpace()
        self._random_sampler = RandomSampler(seed=seed)
        self._population_size = population_size
        self._num_weight_vectors = num_weight_vectors
        self._num_decomposition = num_decomposition
        self._scalar_aggregation_function = scalar_aggregation_function

        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        self._child_generation_strategy = MOEAdChildGenerationStrategy(
            population_size=self._population_size,
            scalar_aggregation_function=self._scalar_aggregation_function,
        )

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        parent_generation, parent_population = self._collect_parent_population(study)

        generation = parent_generation + 1
        study._storage.set_trial_system_attr(
            trial._trial_id, _GENERATION_KEY, generation
        )

        if parent_generation < 0:
            return {}

        return self._child_generation_strategy(study, search_space, parent_population)

    def sample_independent(self, study, trial, param_name, param_distribution) -> Any:
        return self._random_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        return self._random_sampler.reseed_rng()

    def _collect_parent_population(self, study: Study) -> tuple[int, list[FrozenTrial]]:
        trials = study.get_trials(deepcopy=False)

        generation_to_runnings = defaultdict(list)
        generation_to_population = defaultdict(list)
        for trial in trials:
            if _GENERATION_KEY not in trial.system_attrs:
                continue

            generation = trial.system_attrs[_GENERATION_KEY]
            if trial.state != optuna.trial.TrialState.COMPLETE:
                if trial.state == optuna.trial.TrialState.RUNNING:
                    generation_to_runnings[generation].append(trial)
                continue

            generation_to_population[generation].append(trial)

        hasher = hashlib.sha256()
        parent_population: list[FrozenTrial] = []
        parent_generation = -1
        while True:
            generation = parent_generation + 1
            population = generation_to_population[generation]

            if len(population) < self._population_size:
                break

            for trial in generation_to_runnings[generation]:
                hasher.update(bytes(str(trial.number), "utf-8"))

            cache_key = "{}:{}".format(_POPULATION_CACHE_KEY_PREFIX, hasher.hexdigest())
            study_system_attrs = study._storage.get_study_system_attrs(study._study_id)
            cached_generation, cached_population_numbers = study_system_attrs.get(
                cache_key, (-1, [])
            )
            if cached_generation >= generation:
                generation = cached_generation
                population = [trials[n] for n in cached_population_numbers]
            else:
                population.extend(parent_population)

                if len(generation_to_runnings[generation]) == 0:
                    population_numbers = [t.number for t in population]
                    study._storage.set_study_system_attr(
                        study._study_id, cache_key, (generation, population_numbers)
                    )

            parent_generation = generation
            parent_population = population

        return parent_generation, parent_population


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
        population_size: int,
        scalar_aggregation_function: str,
    ) -> None:
        self._population_size = population_size
        self._scalar_aggregation_function = scalar_aggregation_function

    def __call__(self, study, search_space, parent_population) -> dict[str, Any]:
        return {}
