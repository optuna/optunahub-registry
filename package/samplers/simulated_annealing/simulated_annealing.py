from __future__ import annotations

from typing import Any

import numpy as np
import optuna


class SimulatedAnnealingSampler(optuna.samplers.BaseSampler):
    """Sampler based on Simulated Annealing algorithm.

    Args:
        temperature (int):
            Temperature for annealing.
    """

    def __init__(self, temperature: float = 100) -> None:
        self._rng = np.random.RandomState()
        self._temperature = temperature  # Current temperature.
        self._current_trial = None  # Current state.

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        # Simulated Annealing algorithm.
        # 1. Calculate transition probability.
        prev_trial = study.trials[-2]
        if self._current_trial is None or prev_trial.value <= self._current_trial.value:
            probability = 1.0
        else:
            probability = np.exp(
                (self._current_trial.value - prev_trial.value) / self._temperature
            )
        self._temperature *= 0.9  # Decrease temperature.

        # 2. Transit the current state if the previous result is accepted.
        if self._rng.uniform(0, 1) < probability:
            self._current_trial = prev_trial

        # 3. Sample parameters from the neighborhood of the current point.
        # The sampled parameters will be used during the next execution of
        # the objective function passed to the study.
        params: dict[str, Any] = {}
        for param_name, param_distribution in search_space.items():
            if (
                not isinstance(param_distribution, optuna.distributions.FloatDistribution)
                or (param_distribution.step is not None and param_distribution.step != 1)
                or param_distribution.log
            ):
                msg = (
                    "Only suggest_float() with `step` `None` or 1.0 and"
                    " `log` `False` is supported"
                )
                raise NotImplementedError(msg)

            assert self._current_trial is not None
            current_value = self._current_trial.params[param_name]
            width = (param_distribution.high - param_distribution.low) * 0.1
            neighbor_low = max(current_value - width, param_distribution.low)
            neighbor_high = min(current_value + width, param_distribution.high)
            params[param_name] = self._rng.uniform(neighbor_low, neighbor_high)

        return params

    # The rest are unrelated to SA algorithm: boilerplate
    def infer_relative_search_space(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)
