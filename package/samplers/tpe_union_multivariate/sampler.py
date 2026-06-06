from __future__ import annotations

import json
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from .parzen_estimator import _ParzenEstimator
from .parzen_estimator import _ParzenEstimatorParameters
from .union import UnionSearchSpace


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.study import Study

_RELATIVE_PARAMS_KEY = "tpe:relative_params"
_SYSTEM_ATTR_MAX_LENGTH = 2045


def _default_weights(x: int) -> np.ndarray:
    if x == 0:
        return np.asarray([])
    return (
        np.ones(x)
        if x < 25
        else np.concatenate([np.linspace(1.0 / x, 1.0, num=x - 25), np.ones(25)], axis=0)
    )


def _gamma_func(x: int) -> int:
    return min(int(np.ceil(0.1 * x)), 25)


class TPEUnionMultivariateSampler(BaseSampler):
    def __init__(self, n_startup_trials: int = 10, seed: int | None = None) -> None:
        self._n_startup_trials = n_startup_trials
        self._seed = seed
        self._rng = np.random.RandomState(seed)

        self._parzen_estimator_parameters = _ParzenEstimatorParameters(
            prior_weight=1.0,
            consider_magic_clip=True,
            consider_endpoints=False,
            weights=_default_weights,
            multivariate=True,
            categorical_distance_func={},
        )
        self._n_ei_candidates = 24
        self._gamma = _gamma_func
        self._union_search_space = UnionSearchSpace(include_pruned=True)
        self._parzen_estimator_cls = _ParzenEstimator

        self._independent_fallback_sampler = TPESampler(
            n_startup_trials=n_startup_trials,
            seed=seed,
            multivariate=False,
            group=False,
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        pass

    def after_trial(
        self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float] | None
    ) -> None:
        pass

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._union_search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution
        return search_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE, TrialState.PRUNED)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        if len(trials) < self._n_startup_trials:
            return {}

        return self._sample(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_fallback_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _get_params(self, trial: FrozenTrial) -> dict[str, Any]:
        params_strs = []
        i = 0
        while params_str_i := trial.system_attrs.get(f"{_RELATIVE_PARAMS_KEY}:{i}"):
            params_strs.append(params_str_i)
            i += 1
        if len(params_strs) == 0:
            return trial.params
        try:
            params = json.loads("".join(params_strs))
        except json.JSONDecodeError:
            return trial.params
        params.update(trial.params)
        return params

    def _get_internal_repr(
        self, trials: list[FrozenTrial], search_space: dict[str, BaseDistribution]
    ) -> dict[str, np.ndarray]:
        values: dict[str, list[float]] = {param_name: [] for param_name in search_space}
        for trial in trials:
            params = self._get_params(trial)
            for param_name, distribution in search_space.items():
                if param_name in params:
                    values[param_name].append(distribution.to_internal_repr(params[param_name]))
                else:
                    values[param_name].append(np.nan)
        return {k: np.asarray(v) for k, v in values.items()}

    def _sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        trials = study._get_trials(
            deepcopy=False, states=[TrialState.COMPLETE, TrialState.PRUNED], use_cache=True
        )

        n = len(trials)
        n_below = self._gamma(n)

        if study.direction == optuna.study.StudyDirection.MINIMIZE:
            sorted_trials = sorted(
                trials, key=lambda t: cast(float, t.value if t.value is not None else float("inf"))
            )
        else:
            sorted_trials = sorted(
                trials,
                key=lambda t: cast(float, t.value if t.value is not None else float("-inf")),
                reverse=True,
            )

        below_trials = sorted_trials[:n_below]
        above_trials = sorted_trials[n_below:]

        observations_below = self._get_internal_repr(below_trials, search_space)
        observations_above = self._get_internal_repr(above_trials, search_space)

        mpe_below = self._parzen_estimator_cls(
            observations_below, search_space, self._parzen_estimator_parameters
        )
        mpe_above = self._parzen_estimator_cls(
            observations_above, search_space, self._parzen_estimator_parameters
        )

        samples_below = mpe_below.sample(self._rng, self._n_ei_candidates)

        log_likelihoods_below = mpe_below.log_pdf(samples_below)
        log_likelihoods_above = mpe_above.log_pdf(samples_below)
        acquisition_func_vals = log_likelihoods_below - log_likelihoods_above

        best_idx = np.argmax(acquisition_func_vals)
        ret = {k: v[best_idx].item() for k, v in samples_below.items()}

        for param_name, dist in search_space.items():
            if np.isnan(ret[param_name]):
                ret[param_name] = self._independent_fallback_sampler.sample_independent(
                    study, trial, param_name, dist
                )
            else:
                ret[param_name] = dist.to_external_repr(ret[param_name])
        return ret
