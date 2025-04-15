from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
import hashlib
from typing import Any
from typing import TYPE_CHECKING

import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers import NSGAIISampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from ._child_generation_strategy import NSGAIIwITChildGenerationStrategy
from ._mutations._base import BaseMutation
from ._mutations._uniform import UniformMutation


if TYPE_CHECKING:
    from optuna.study import Study


# Define key names of `Trial.system_attrs`.
_GENERATION_KEY = "nsga2wit:generation"
_POPULATION_CACHE_KEY_PREFIX = "nsga2wit:population"


class NSGAIIwITSampler(NSGAIISampler):
    def __init__(
        self,
        *,
        population_size: int = 50,
        mutation: BaseMutation | None = None,
        mutation_prob: float | None = None,
        crossover: BaseCrossover | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: int | None = None,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        elite_population_selection_strategy: (
            Callable[[Study, list[FrozenTrial]], list[FrozenTrial]] | None
        ) = None,
        after_trial_strategy: (
            Callable[[Study, FrozenTrial, TrialState, Sequence[float] | None], None] | None
        ) = None,
    ) -> None:
        if mutation is None:
            mutation = UniformMutation()

        if crossover is None:
            crossover = UniformCrossover(swapping_prob)

        super().__init__(
            population_size=population_size,
            mutation_prob=mutation_prob,
            crossover=crossover,
            crossover_prob=crossover_prob,
            swapping_prob=swapping_prob,
            seed=seed,
            constraints_func=constraints_func,
            elite_population_selection_strategy=elite_population_selection_strategy,
            child_generation_strategy=NSGAIIwITChildGenerationStrategy(
                mutation=mutation,
                mutation_prob=mutation_prob,
                crossover=crossover,
                crossover_prob=crossover_prob,
                swapping_prob=swapping_prob,
                constraints_func=constraints_func,
                rng=LazyRandomState(seed),
            ),
            after_trial_strategy=after_trial_strategy,
        )

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        parent_generation, parent_population = self._collect_parent_population(study)

        generation = parent_generation + 1
        study._storage.set_trial_system_attr(trial._trial_id, _GENERATION_KEY, generation)

        if parent_generation < 0:
            return {}

        return self._child_generation_strategy(study, search_space, parent_population)

    def _collect_parent_population(self, study: Study) -> tuple[int, list[FrozenTrial]]:
        trials = study._get_trials(deepcopy=False, use_cache=True)

        generation_to_runnings = defaultdict(list)
        generation_to_population = defaultdict(list)
        for trial in trials:
            if _GENERATION_KEY not in trial.system_attrs:
                generation = 0
            else:
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
