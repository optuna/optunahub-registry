from __future__ import annotations

import binascii
import math
import numpy as np
from typing import Any

from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.pruners import SuccessiveHalvingPruner
from optuna.pruners._successive_halving import _get_current_rung
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.samplers import IntersectionSearchSpace
from optuna.study import Study
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
        print(f"sample_relative: iteration_id = {iteration_id}, bracket_id = {bracket_id}, budget_id = {budget_id}")

        subpopulation: list[FrozenTrial] = []

        if bracket_id == 0:
            subpopulation.extend([t for t in trials if self._get_iteration_id(study, t) == iteration_id - 1 and self._get_bracket_id(study, t) == budget_id and self._get_budget_id(study, t, bracket_id) == budget_id])
        else:
            subpopulation.extend([t for t in trials if self._get_iteration_id(study, t) == iteration_id and self._get_bracket_id(study, t) == bracket_id - 1 and self._get_budget_id(study, t, bracket_id) == budget_id])
        
        if budget_id > bracket_id:
            previous_budget_trials = [t for t in trials if self._get_iteration_id(study, t) == iteration_id and self._get_bracket_id(study, t) == bracket_id and self._get_budget_id(study, t, bracket_id) == budget_id - 1]
            reduction_factor = study.pruner._pruners[0]._reduction_factor
            promotable_trials = sorted(previous_budget_trials, key=lambda t: t.intermediate_values[t.last_step])[:len(previous_budget_trials) // reduction_factor]
            subpopulation.extend(promotable_trials)

        # print(f"sample_relative: subpopulation = {[t.number for t in subpopulation]}")

        if len(subpopulation) < 3:
            print(f"len(subpopulation) = {len(subpopulation)}, extending subpopulation with trials from all budgets.")
            subpopulation.extend(trials)

        # Select parents according to the roulette wheel selection.    
        parent0, parent1, parent2 = self._select_parents(subpopulation)

        # print(f"selected parents: {parent0.number}, {parent1.number}, {parent2.number}")
        # Mutate selected parents using rand/1 mutation strategy.
        mutant = self._mutate(parent0, parent1, parent2, search_space)

        # Crossover mutant based on binomial crossover.
        offspring = self._crossover(mutant, parent0, search_space)
        # print(f"sample_relative: offspring = {offspring}")

        return offspring
    
    def sample_independent(
        self, study: Study, trial: FrozenTrial, param_name: str, param_distribution: BaseDistribution
    ) -> Any:
        ret = self._random_sampler.sample_independent(study, trial, param_name, param_distribution)
        print(f"sample_independent: {param_name} = {ret}")
        return ret

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # `cma` cannot handle distributions that contain just a single value, so we skip
                # them. Note that the parameter values for such distributions are sampled in
                # `Trial`.
                continue
            search_space[name] = distribution

        return search_space

    def _get_n_trials_in_first_dehb_iteration(self, study: Study) -> int:
        dehb_pruner = study.pruner
        assert isinstance(dehb_pruner, DEHBPruner)

        if dehb_pruner._n_brackets is None:
            return 2 ** 32 - 1  # Return a large number.

        s_max = dehb_pruner._n_brackets - 1
   
        successive_halving_pruner = dehb_pruner._pruners[0]
        assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)

        assert isinstance(successive_halving_pruner._min_resource, int)
        min_resource = successive_halving_pruner._min_resource
        reduction_factor = successive_halving_pruner._reduction_factor

        n = 0
        for i in range(s_max + 1):
            for j in range(i, s_max + 1):
                n += min_resource * reduction_factor ** (s_max - j)
        return n
    
    def _get_iteration_id(self, study: Study, trial: FrozenTrial) -> int:
        n_trials_per_iteration = self._get_n_trials_in_first_dehb_iteration(study)
        return trial.number // n_trials_per_iteration

    def _get_bracket_id(self, study: Study, trial: FrozenTrial) -> int:
        dehb_pruner = study.pruner
        assert isinstance(dehb_pruner, DEHBPruner)
        return dehb_pruner._get_bracket_id_after_init(study, trial)
        # # if trial.state == TrialState.RUNNING:
        # #     self = study.pruner
        # #     assert isinstance(self, DEHBPruner)
        # #     return self._get_bracket_id_after_init_after_init(study, trial)
        # assert isinstance(dehb_pruner._n_brackets, int)
        # s_max = dehb_pruner._n_brackets - 1

        # successive_halving_pruner = dehb_pruner._pruners[0]
        # assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)

        # assert isinstance(successive_halving_pruner._min_resource, int)
        # min_resource = successive_halving_pruner._min_resource
        # reduction_factor = successive_halving_pruner._reduction_factor
        
        # n_trials_for_each_bracket = []
        # for i in range(s_max + 1):
        #     n = 0
        #     for j in range(i, s_max + 1):
        #         n += reduction_factor ** (s_max - j)
        #     n_trials_for_each_bracket.append(n)

        # n_trials_per_iteration = self._get_n_trials_in_first_dehb_iteration(study)
        # trial_number_in_iteration = trial.number % n_trials_per_iteration
        # for i in range(s_max + 1):
        #     if trial_number_in_iteration < n_trials_for_each_bracket[i]:
        #         return i
        #     trial_number_in_iteration -= n_trials_for_each_bracket[i]
        
        # assert False, "This line should never be reached."


    def _get_budget_id(self, study: Study, trial: FrozenTrial, bracket_id: int) -> int:
        dehb_pruner = study.pruner
        assert isinstance(dehb_pruner, DEHBPruner)
        return dehb_pruner._get_budget_id(study, trial, bracket_id)

    def _select_parents(self, subpopulation: list[FrozenTrial]) -> tuple[FrozenTrial, FrozenTrial, FrozenTrial]:
        values = [t.value for t in subpopulation]
        idxs = self._rng.rng.choice(len(subpopulation), 3, replace=False, p=values/np.sum(values))
        return subpopulation[idxs[0]], subpopulation[idxs[1]], subpopulation[idxs[2]]
    
    def _mutate(self, parent0: FrozenTrial, parent1: FrozenTrial, parent2: FrozenTrial, search_space: dict[str, BaseDistribution]) -> dict[str, Any]:
        _transform = _SearchSpaceTransform(search_space)
        
        params0 = _transform.transform(parent0.params)
        params1 = _transform.transform(parent1.params)
        params2 = _transform.transform(parent2.params)

        mutant_params = params0 + self._scaling_factor * (params1 - params2)
        mutant = _transform.untransform(mutant_params)

        return mutant
    
    def _crossover(self, mutant: dict[str, Any], parent0: FrozenTrial, search_space: dict[str, BaseDistribution]) -> dict[str, Any]:
        rand_idx = self._rng.rng.randint(0, len(search_space))
        offspring = {}
        for i, (name, distribution) in enumerate(search_space.items()):
            if i == rand_idx or self._rng.rng.rand() < self._crossover_prob:
                offspring[name] = mutant[name]
            else:
                offspring[name] = parent0.params[name]
        return offspring
        