from __future__ import annotations

import math

import numpy as np
from optuna import logging
from optuna.pruners import HyperbandPruner
from optuna.pruners import SuccessiveHalvingPruner
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = logging.get_logger(__name__)


class DEHBPruner(HyperbandPruner):
    def __init__(
        self,
        min_resource: int = 1,
        max_resource: str | int = "auto",
        reduction_factor: int = 3,
        bootstrap_count: int = 0,
    ) -> None:
        super().__init__(min_resource, max_resource, reduction_factor, bootstrap_count)

        self._budget_candidates: list[int] | None = None

    def prune(self, study: Study, trial: FrozenTrial) -> bool:
        if len(self._pruners) == 0:
            self._try_initialization(study)
            if len(self._pruners) == 0:
                return False

        if self._get_iteration_id(study, trial) == 0:
            if self._prune_for_initial_iteration(study, trial):
                print(f"Pruned trial {trial.number} for initial bracket in step {trial.last_step}")
                return True
            return False

        bracket_id = self._get_bracket_id_after_init(study, trial)
        _logger.debug("{}th bracket is selected".format(bracket_id))
        bracket_study = self._create_bracket_study(study, bracket_id)
        return self._pruners[bracket_id].prune(bracket_study, trial)

    def _prune_for_initial_iteration(self, study: Study, trial: FrozenTrial) -> bool:
        if self._budget_candidates is None:
            self._create_budget_candidates(study)
        assert self._budget_candidates is not None

        # trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED))
        # budget_to_trial_numbers = {budget: [] for budget in self._budget_candidates}
        # for t in trials:
        #     last_step = t.last_step
            
        #     if last_step is None:
        #         continue
        #     assert last_step is not None
        
        #     last_step = self._round_budget(last_step)
        #     budget_to_trial_numbers[last_step].append(t.number)

        # max_budget = max(budget_to_trial_numbers.keys())
        # last_step = trial.last_step
        # if last_step is None:
        #     return False
        # for budget in budget_to_trial_numbers:
        #     if len(budget_to_trial_numbers[budget]) < max_budget / budget:
        #         if last_step < budget:
        #             return False
        #         return True
        # return False
        

        if self._get_iteration_id(study, trial) > 0:
            return False

        
        
        bracket_id = self._get_bracket_id_after_init(study, trial)
        budget_id = self._get_budget_id(study, trial, bracket_id)
        budget = self._budget_candidates[budget_id]
        last_step = trial.last_step
        if last_step is None:
            return False
        if last_step < budget:
            return False
        # print(f"Budget candidates: {self._budget_candidates}")
        # print(f"Total number of trials in the first DEHB iteration: {self._get_n_trials_in_first_dehb_iteration(study)}")
        # print(f"trial.number: {trial.number}, iteration_id: {self._get_iteration_id(study, trial)}, bracket_id: {bracket_id}, budget_id: {budget_id}")

        return True

    def _create_budget_candidates(self, study: Study) -> None:
        assert self._n_brackets is not None
        successive_halving_pruner = self._pruners[0]
        assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)
        min_resource = successive_halving_pruner._min_resource
        reduction_factor = successive_halving_pruner._reduction_factor

        budget_candidates = []
        for i in range(self._n_brackets):
            budget_candidates.append(min_resource * reduction_factor ** i)
        self._budget_candidates = budget_candidates

    def _get_n_trials_in_first_dehb_iteration(self, study: Study) -> int:
        if self._n_brackets is None:
            return 2 ** 32 - 1  # Return a large number.

        s_max = self._n_brackets - 1
   
        successive_halving_pruner = self._pruners[0]
        assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)

        assert isinstance(successive_halving_pruner._min_resource, int)
        min_resource = successive_halving_pruner._min_resource
        reduction_factor = successive_halving_pruner._reduction_factor

        n = 0
        for i in range(s_max + 1):
            for j in range(i, s_max + 1):
                n += reduction_factor ** (s_max - j)
        return n
    
    def _get_iteration_id(self, study: Study, trial: FrozenTrial) -> int:
        n_trials_per_iteration = self._get_n_trials_in_first_dehb_iteration(study)
        return trial.number // n_trials_per_iteration

    def _get_bracket_id_after_init(self, study: Study, trial: FrozenTrial) -> int:
        # if trial.state == TrialState.RUNNING:
        #     self = study.pruner
        #     assert isinstance(self, DEHBPruner)
        #     return self._get_bracket_id_after_init_after_init(study, trial)
        assert isinstance(self._n_brackets, int)
        s_max = self._n_brackets - 1

        successive_halving_pruner = self._pruners[0]
        assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)

        assert isinstance(successive_halving_pruner._min_resource, int)
        min_resource = successive_halving_pruner._min_resource
        reduction_factor = successive_halving_pruner._reduction_factor
        
        n_trials_for_each_bracket = []
        for i in range(s_max + 1):
            n = 0
            for j in range(i, s_max + 1):
                n += reduction_factor ** (s_max - j)
            n_trials_for_each_bracket.append(n)

        n_trials_per_iteration = self._get_n_trials_in_first_dehb_iteration(study)
        trial_number_in_iteration = trial.number % n_trials_per_iteration
        for i in range(s_max + 1):
            if trial_number_in_iteration < n_trials_for_each_bracket[i]:
                return i
            trial_number_in_iteration -= n_trials_for_each_bracket[i]
        
        assert False, "This line should never be reached."

    def _get_budget_id(self, study: Study, trial: FrozenTrial, bracket_id: int) -> int:
        budget_candidates = self._budget_candidates
        assert budget_candidates is not None
        assert isinstance(self._n_brackets, int)
        s_max = self._n_brackets - 1

        successive_halving_pruner = self._pruners[0]
        assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)

        assert isinstance(successive_halving_pruner._min_resource, int)
        min_resource = successive_halving_pruner._min_resource
        reduction_factor = successive_halving_pruner._reduction_factor
        
        if trial.state != TrialState.RUNNING:
            last_step = trial.last_step
            assert last_step is not None
            
            successive_halving_pruner = self._pruners[0]
            assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)
            reduction_factor = successive_halving_pruner._reduction_factor

            difference = [abs(math.log(x / last_step, reduction_factor)) for x in budget_candidates]
            return np.argmin(difference)

        # probabilities = np.asarray([1 / budget for budget in budget_candidates[bracket_id:]])
        # probabilities /= np.sum(probabilities)
        # probabilities = np.concatenate([np.zeros(bracket_id), probabilities])
        # return self._rng.rng.choice(len(budget_candidates), p=probabilities)
        n_trials_for_each_bracket = []
        for i in range(s_max + 1):
            n = 0
            for j in range(i, s_max + 1):
                n += reduction_factor ** (s_max - j)
            n_trials_for_each_bracket.append(n)

        n_trials_in_the_bracket = []
        for j in range(bracket_id, s_max + 1):
            n = reduction_factor ** (s_max - j)
            n_trials_in_the_bracket.append(n)

        n_trials_per_iteration = self._get_n_trials_in_first_dehb_iteration(study)
        trial_number_in_iteration = trial.number % n_trials_per_iteration
        print(f"trial_number_in_iteration: {trial_number_in_iteration}")
        for i in range(0, bracket_id):
            trial_number_in_iteration -= n_trials_for_each_bracket[i]

        for j in range(bracket_id, s_max + 1):
            if trial_number_in_iteration < n_trials_in_the_bracket[j - bracket_id]:
                return j
            trial_number_in_iteration -= n_trials_in_the_bracket[j - bracket_id]
     
        assert False, "This line should never be reached."

    # def _round_budget(self, budget: int) -> int:
    #     assert self._budget_candidates is not None

    #     budget_candidates = self._budget_candidates
    #     successive_halving_pruner = self._pruners[0]
    #     assert isinstance(successive_halving_pruner, SuccessiveHalvingPruner)
    #     reduction_factor = successive_halving_pruner._reduction_factor

    #     return min(budget_candidates, key=lambda x: abs(math.log(x /budget, reduction_factor)))
