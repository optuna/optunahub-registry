from __future__ import annotations

from typing import Any

import numpy as np
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import IntersectionSearchSpace
from optuna.samplers import RandomSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from .pruner import DEHBPruner


class DEHBSampler(BaseSampler):
    def __init__(
        self,
        scaling_factor: float = 0.5,
        crossover_prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self._scaling_factor = scaling_factor
        self._crossover_prob = crossover_prob
        self._random_sampler = RandomSampler(seed=seed)
        self._rng = LazyRandomState(seed)
        self._search_space = IntersectionSearchSpace()

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if len(search_space) == 0:
            return {}

        trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED))

        if len(trials) <= self._get_n_trials_in_first_dehb_iteration(study):
            return {}

        iteration_id = self._get_iteration_id(study, trial)
        bracket_id = self._get_bracket_id(study, trial)
        budget_id = self._get_budget_id(study, trial, bracket_id)
        assert iteration_id > 0

        subpopulation: list[FrozenTrial] = []

        if bracket_id == 0:
            subpopulation.extend(
                [
                    t
                    for t in trials
                    if self._get_iteration_id(study, t) == iteration_id - 1
                    and self._get_bracket_id(study, t) == budget_id
                    and self._get_budget_id(study, t, bracket_id) == budget_id
                ]
            )
        else:
            subpopulation.extend(
                [
                    t
                    for t in trials
                    if self._get_iteration_id(study, t) == iteration_id
                    and self._get_bracket_id(study, t) == bracket_id - 1
                    and self._get_budget_id(study, t, bracket_id) == budget_id
                ]
            )

        if budget_id > bracket_id:
            previous_budget_trials = [
                t
                for t in trials
                if self._get_iteration_id(study, t) == iteration_id
                and self._get_bracket_id(study, t) == bracket_id
                and self._get_budget_id(study, t, bracket_id) == budget_id - 1
            ]
            reduction_factor = study.pruner._pruners[0]._reduction_factor
            promotable_trials = sorted(
                previous_budget_trials, key=lambda t: t.intermediate_values[t.last_step]
            )[: len(previous_budget_trials) // reduction_factor]
            subpopulation.extend(promotable_trials)

        if len(subpopulation) < 3:
            subpopulation.extend(trials)

        # Select parents according to the roulette wheel selection.
        parent0, parent1, parent2 = self._select_parents(subpopulation, study.direction)

        # Mutate selected parents using rand/1 mutation strategy.
        mutant = self._mutate(parent0, parent1, parent2, search_space)

        # Crossover mutant based on binomial crossover.
        offspring = self._crossover(mutant, parent0, search_space)

        return offspring

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        ret = self._random_sampler.sample_independent(study, trial, param_name, param_distribution)
        return ret

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def _get_n_trials_in_first_dehb_iteration(self, study: Study) -> int:
        dehb_pruner = study.pruner
        assert isinstance(dehb_pruner, DEHBPruner)
        return dehb_pruner._get_n_trials_in_first_dehb_iteration(study)

    def _get_iteration_id(self, study: Study, trial: FrozenTrial) -> int:
        dehb_pruner = study.pruner
        assert isinstance(dehb_pruner, DEHBPruner)
        return dehb_pruner._get_iteration_id(study, trial)

    def _get_bracket_id(self, study: Study, trial: FrozenTrial) -> int:
        dehb_pruner = study.pruner
        assert isinstance(dehb_pruner, DEHBPruner)
        return dehb_pruner._get_bracket_id_after_init(study, trial)

    def _get_budget_id(self, study: Study, trial: FrozenTrial, bracket_id: int) -> int:
        dehb_pruner = study.pruner
        assert isinstance(dehb_pruner, DEHBPruner)
        return dehb_pruner._get_budget_id(study, trial, bracket_id)

    def _select_parents(
        self, subpopulation: list[FrozenTrial], direction: StudyDirection
    ) -> tuple[FrozenTrial, FrozenTrial, FrozenTrial]:
        sign = 1 if direction == StudyDirection.MAXIMIZE else -1
        values = np.array([sign * t.value for t in subpopulation])
        values = (
            (values - values.min()) / (values.max() - values.min())
            if values.max() != values.min()
            else np.ones_like(values)
        )
        probabilities = values / values.sum()
        idxs = self._rng.rng.choice(len(subpopulation), 3, p=probabilities)
        return subpopulation[idxs[0]], subpopulation[idxs[1]], subpopulation[idxs[2]]

    def _mutate(
        self,
        parent0: FrozenTrial,
        parent1: FrozenTrial,
        parent2: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        _transform = _SearchSpaceTransform(search_space)

        params0 = _transform.transform(parent0.params)
        params1 = _transform.transform(parent1.params)
        params2 = _transform.transform(parent2.params)

        mutant_params = params0 + self._scaling_factor * (params1 - params2)
        mutant = _transform.untransform(mutant_params)

        return mutant

    def _crossover(
        self,
        mutant: dict[str, Any],
        parent0: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        rand_idx = self._rng.rng.randint(0, len(search_space))
        offspring = {}
        for i, (name, distribution) in enumerate(search_space.items()):
            if i == rand_idx or self._rng.rng.rand() < self._crossover_prob:
                offspring[name] = mutant[name]
            else:
                offspring[name] = parent0.params[name]
        return offspring
