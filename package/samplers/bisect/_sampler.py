from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from optuna import Study
    from optuna import Trial
    from optuna.distributions import BaseDistribution
    from optuna.distributions import FloatDistribution
    from optuna.distributions import IntDistribution
    from optuna.trial import FrozenTrial


PREFIX_LEFT = "bisect:left_"
PREFIX_RIGHT = "bisect:right_"


class BisectSampler(optuna.samplers.BaseSampler):
    """Sampler using bisect (binary search) indepedently for each parameter.

    Args:
        rtol:
            The relative tolerance parameter to be used to judge whether all the parameters are
            converged. Default to that in `np.isclose`, i.e., 1e-5.
        atol:
            The absolute tolerance parameter to be used to judge whether all the parameters are
            converged. Default to that in `np.isclose`, i.e., 1e-8.
    """

    def __init__(self, *, rtol: float = 1e-5, atol: float = 1e-8) -> None:
        self._atol = atol
        self._rtol = rtol
        self._search_space: dict[str, IntDistribution | FloatDistribution] = {}
        self._stop_flag = False

    def infer_relative_search_space(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return {}

    def _get_current_left_and_right(
        self, study: Study, param_name: str
    ) -> tuple[int | float, int | float]:
        left_key = f"{PREFIX_LEFT}{param_name}"
        right_key = f"{PREFIX_RIGHT}{param_name}"
        system_attrs = study._storage.get_study_system_attrs(study._study_id)
        left = system_attrs[left_key]
        right = system_attrs[right_key]
        assert isinstance(left, (int, float)) and isinstance(right, (int, float))
        return left, right

    def _set_left_and_right(
        self,
        study: Study,
        param_name: str,
        *,
        left: int | float | None = None,
        right: int | float | None = None,
    ) -> None:
        left_key = f"{PREFIX_LEFT}{param_name}"
        right_key = f"{PREFIX_RIGHT}{param_name}"
        if left is not None:
            study._storage.set_study_system_attr(study._study_id, left_key, left)
        if right is not None:
            study._storage.set_study_system_attr(study._study_id, right_key, right)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
            raise ValueError("CategoricalDistribution is not supported.")
        if param_distribution.log:
            raise ValueError("Log scale is not supported.")

        self._search_space[param_name] = param_distribution
        step = param_distribution.step
        is_discrete = step is not None
        if trial.number == 0:
            right = param_distribution.high + step if is_discrete else param_distribution.high
            self._set_left_and_right(study, param_name, left=param_distribution.low, right=right)

        left, right = self._get_current_left_and_right(study, param_name)
        if not is_discrete:
            mid = (left + right) / 2.0
            return mid

        possible_param_values = self._get_possible_param_values(param_distribution)
        indices = np.arange(len(possible_param_values))
        left_index = indices[np.isclose(possible_param_values, left)][0]
        right_index = indices[np.isclose(possible_param_values, right)][0]
        mid_index = (right_index + left_index) // 2
        assert mid_index != len(possible_param_values) - 1, "The last element is for convenience."
        return possible_param_values[mid_index].item()

    def _get_possible_param_values(
        self, param_distribution: FloatDistribution | IntDistribution
    ) -> np.ndarray:
        step = param_distribution.step
        low = param_distribution.low
        # The last element is padded to code the binary search routine cleaner.
        high = param_distribution.high + step
        assert step is not None
        n_steps = int(np.round((high - low) / step)) + 1
        return np.linspace(low, high, n_steps)

    def _is_param_converged(self, study: Study, param_name: str) -> bool:
        left, right = self._get_current_left_and_right(study, param_name)
        dist = self._search_space[param_name]
        is_discrete = dist.step is not None
        if not is_discrete:
            return math.isclose(left, right, abs_tol=self._atol, rel_tol=self._rtol)

        possible_param_values = self._get_possible_param_values(dist)
        indices = np.arange(len(possible_param_values))
        left_index = indices[np.isclose(possible_param_values, left)][0]
        right_index = indices[np.isclose(possible_param_values, right)][0]
        return right_index - left_index <= 1

    @staticmethod
    def score_func(trial: Trial | FrozenTrial) -> float:
        score = 0.0
        for k, param_value in trial.params.items():
            low = trial.distributions[k].low
            high = trial.distributions[k].high
            is_too_high = trial.user_attrs[f"{k}_is_too_high"]
            score += (2 * is_too_high - 1) * (param_value - low) / (high - low)

        return score

    @staticmethod
    def get_best_param(study: Study) -> dict[str, Any]:
        best_param: dict[str, Any] = {}
        for t in study.trials:
            params = t.params
            user_attrs = t.user_attrs
            for k, v in params.items():
                if user_attrs[f"{k}_is_too_high"]:
                    continue

                best_param[k] = max(v, best_param[k]) if k in best_param else v

        return best_param

    def _enqueue_best_param(self, study: Study) -> None:
        study.enqueue_trial(self.get_best_param(study))
        self._stop_flag = True

    def after_trial(
        self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float] | None
    ) -> None:
        if values is None or state != TrialState.COMPLETE:
            return

        converged = True
        for param_name, param_value in trial.params.items():
            is_too_high_key = f"{param_name}_is_too_high"
            too_high = trial.user_attrs.get(is_too_high_key)
            if too_high is None or not isinstance(too_high, bool):
                raise ValueError(
                    f"BisectSampler requires an attribute to judge whether each param is too high."
                    f' Set it via `trial.set_user_attr("{is_too_high_key}", <True or False>)`.'
                )

            if too_high:  # param is too high.
                self._set_left_and_right(study, param_name, right=param_value)
            else:  # param is too low.
                self._set_left_and_right(study, param_name, left=param_value)
            converged &= self._is_param_converged(study, param_name)

        if self._stop_flag:
            study.stop()
        if converged and not self._stop_flag:
            self._enqueue_best_param(study)

        if not math.isclose(self.score_func(trial), values[0]):
            expected_value = self.score_func(trial)
            got_value = values[0]
            raise ValueError(
                "Please return `BisectSampler.score_func(trial)` in your objective. "
                f"Expected {expected_value}, but got {got_value}"
            )
