from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any
from typing import Dict
from typing import TYPE_CHECKING

from _child_generation_strategy import MOEAdChildGenerationStrategy
from _elite_population_selection_strategy import MOEAdElitePopulationSelectionStrategy
import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import FrozenTrial


if TYPE_CHECKING:
    from optuna.study import Study

# Define key names of `Trial.system_attrs`.
_GENERATION_KEY = "moead:generation"
_POPULATION_CACHE_KEY_PREFIX = "moead:population"


class MOEAdSampler(BaseSampler):
    def __init__(
        self,
        *,
        population_size: int = 100,
        n_neighbors: int | None = None,
        scalar_aggregation_func: str = "tchebycheff",
        mutation_prob: float | None = None,
        crossover: BaseCrossover | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Multi-objective sampler using the MOEA/D algorithm.

        MOEA/D stands for "Multi-Objective Evolutionary Algorithm based on Decomposition.

        For more information about MOEA/D, please refer to the following paper:

        - `MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition<https://doi.org/10.1109/TEVC.2007.892759>`__

        Args:
            seed:
                Seed for random number generator.
            population_size:
                The number of individuals in the population.
            T:
                The number of the weight vectors in the neighborhood of each weight vector.
            scalar_aggregation_function:
                The scalar aggregation function to use. The default is "weighted_sum". Other options are "tchebycheff" and "PBI".
        """
        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        if n_neighbors is None:
            n_neighbors = population_size // 10
        elif n_neighbors >= population_size:
            raise ValueError("`T` must be less than `population_size`.")

        if scalar_aggregation_func not in ["weighted_sum", "tchebycheff", "PBI"]:
            raise ValueError(
                "`scalar_aggregation_function` must be one of 'weighted_sum', 'tchebycheff', 'PBI'."
            )

        if crossover is None:
            crossover = UniformCrossover(swapping_prob)
        self._population_size = population_size
        self._random_sampler = RandomSampler(seed=seed)
        self._rng = LazyRandomState(seed)
        self._search_space = IntersectionSearchSpace()
        self._seed = seed
        self._weight_vectors = None

        self._elite_population_selection_strategy = MOEAdElitePopulationSelectionStrategy(
            seed=seed,
            population_size=population_size,
            n_neighbors=n_neighbors,
            scalar_aggregation_func=scalar_aggregation_func,
        )
        self._child_generation_strategy = MOEAdChildGenerationStrategy(
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            swapping_prob=swapping_prob,
            crossover=crossover,
            rng=self._rng,
        )

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        parent_generation, parent_population = self._collect_parent_population(study)

        generation = parent_generation + 1
        study._storage.set_trial_system_attr(trial._trial_id, _GENERATION_KEY, generation)

        if parent_generation < 0:
            return {}

        neighbors = self._elite_population_selection_strategy._neighbors
        return self._child_generation_strategy(study, search_space, parent_population, neighbors)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # Following parameters are randomly sampled here.
        # 1. A parameter in the initial population/first generation.
        # 2. A parameter to mutate.
        # 3. A parameter excluded from the intersection search space.

        return self._random_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def reseed_rng(self) -> None:
        return self._random_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # The `untransform` method of `optuna._transform._SearchSpaceTransform`
                # does not assume a single value,
                # so single value objects are not sampled with the `sample_relative` method,
                # but with the `sample_independent` method.
                continue
            search_space[name] = distribution
        return search_space

    # This method is same as `optuna.samplers.nsgaii._sampler._collect_parent_population`.
    def _collect_parent_population(self, study: Study) -> tuple[int, list[FrozenTrial]]:
        trials = study._get_trials(deepcopy=False, use_cache=True)

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
                population = self._elite_population_selection_strategy(study, population)

                if len(generation_to_runnings[generation]) == 0:
                    population_numbers = [t.number for t in population]
                    study._storage.set_study_system_attr(
                        study._study_id, cache_key, (generation, population_numbers)
                    )

            parent_generation = generation
            parent_population = population

        return parent_generation, parent_population
