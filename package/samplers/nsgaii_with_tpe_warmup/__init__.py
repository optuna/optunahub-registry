from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from optuna.samplers import NSGAIISampler
from optuna.samplers import TPESampler


if TYPE_CHECKING:
    from optuna import Study
    from optuna.distributions import BaseDistribution
    from optuna.trial import FrozenTrial


MAX_INT32 = (1 << 31) - 1


class NSGAIIWithTPEWarmupSampler(NSGAIISampler):
    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        # Let NSGA-II record the generation, including during the TPE warmup.
        params = super().sample_relative(study, trial, search_space)
        assert self._population_size is not None
        if trial.number < self._population_size:
            seed = self._rng.rng.randint(MAX_INT32)
            sampler = TPESampler(
                multivariate=True, constraints_func=self._constraints_func, seed=seed
            )
            return sampler.sample_relative(study, trial, search_space)

        return params
