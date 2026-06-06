from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


def _calculate_union(
    trials: list[FrozenTrial],
    include_pruned: bool = False,
) -> dict[str, BaseDistribution]:
    search_space: dict[str, BaseDistribution] = {}

    states_of_interest = [TrialState.COMPLETE]
    if include_pruned:
        states_of_interest.append(TrialState.PRUNED)

    for trial in trials:
        if trial.state not in states_of_interest:
            continue
        for param_name, distribution in trial.distributions.items():
            if param_name in search_space:
                if search_space[param_name] != distribution:
                    raise ValueError(
                        f"Inconsistent distributions found for parameter '{param_name}'."
                    )
            else:
                search_space[param_name] = distribution

    return search_space


class UnionSearchSpace:
    def __init__(self, include_pruned: bool = False) -> None:
        self._include_pruned = include_pruned
        self._search_space: dict[str, BaseDistribution] = {}
        self._study_id: int | None = None
        self._cached_trial_number = -1

    def calculate(self, study: Study) -> dict[str, BaseDistribution]:
        if self._study_id is None:
            self._study_id = study._study_id
        elif self._study_id != study._study_id:
            raise ValueError("UnionSearchSpace cannot handle multiple studies.")

        trials = study.get_trials(deepcopy=False)
        if len(trials) == 0:
            return {}

        last_trial_number = trials[-1].number
        if last_trial_number > self._cached_trial_number:
            self._search_space = _calculate_union(trials, self._include_pruned)
            self._cached_trial_number = last_trial_number

        return copy.deepcopy(self._search_space)
