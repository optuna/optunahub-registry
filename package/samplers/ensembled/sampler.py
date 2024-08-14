from __future__ import annotations

from typing import Any
from typing import Sequence

import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class EnsembledSampler(BaseSampler):
    def __init__(self, samplers: list[BaseSampler]) -> None:
        self._samplers = samplers

    def _get_sampler(self, trial: FrozenTrial) -> BaseSampler:
        return self._samplers[trial.number % len(self._samplers)]

    def infer_relative_search_space(
        self, study: optuna.study.Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return self._get_sampler(trial).infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: optuna.Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self._get_sampler(trial).sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._get_sampler(trial).sample_independent(
            study, trial, param_name, param_distribution
        )

    def after_trial(
        self,
        study: optuna.Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._get_sampler(trial).after_trial(study, trial, state, values)
