"""Tests for CatCmawmSampler.

Covers the mapping from ``cmaes.CatCMAwM``'s internal solution representation back to
Optuna's parameter dictionary. A reversed mapping is silently tolerated by Optuna --
``sample_relative()`` may legally return a partial dict -- so these tests assert on the
keys of the returned dict and on whether ``sample_independent()`` is used, rather than
relying on an exception being raised.
"""

from __future__ import annotations

from typing import Any

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import RandomSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
import optunahub
import pytest


CatCmawmSampler = optunahub.load_local_module(
    package="samplers/catcmawm", registry_root="package/"
).CatCmawmSampler


SEARCH_SPACE: dict[str, BaseDistribution] = {
    "x1": FloatDistribution(-5, 5),
    "x2": FloatDistribution(-5, 5),
    "z1": IntDistribution(-1, 1),
    "z2": IntDistribution(-2, 2),
    "c1": CategoricalDistribution([0, 1, 2]),
    "c2": CategoricalDistribution([0, 1, 2]),
}


class _TrackingRandomSampler(RandomSampler):
    """A ``RandomSampler`` that records which parameters it was asked to sample."""

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.calls: list[tuple[int, str]] = []

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self.calls.append((trial.number, param_name))
        return super().sample_independent(study, trial, param_name, param_distribution)


def _objective(trial: optuna.Trial) -> float:
    value = 0.0
    for name, distribution in SEARCH_SPACE.items():
        if isinstance(distribution, FloatDistribution):
            value += trial.suggest_float(name, distribution.low, distribution.high) ** 2
        elif isinstance(distribution, IntDistribution):
            value += trial.suggest_int(name, distribution.low, distribution.high) ** 2
        else:
            assert isinstance(distribution, CategoricalDistribution)
            choice = trial.suggest_categorical(name, distribution.choices)
            assert isinstance(choice, int)
            value += choice
    return value


def test_sample_relative_returns_all_param_names() -> None:
    # The returned dict must be keyed by parameter name, for every parameter type.
    sampler = CatCmawmSampler(search_space=SEARCH_SPACE, seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=1)

    trial = study.ask(SEARCH_SPACE)
    params = sampler.sample_relative(study, trial, SEARCH_SPACE)

    assert set(params.keys()) == set(SEARCH_SPACE.keys())


def test_relative_sampling_is_used_for_all_param_types() -> None:
    # Once the search space is inferred, no parameter should fall back to the
    # independent sampler.
    independent_sampler = _TrackingRandomSampler(seed=0)
    sampler = CatCmawmSampler(seed=0, independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=3)

    fallbacks_after_first_trial = [c for c in independent_sampler.calls if c[0] > 0]
    assert fallbacks_after_first_trial == []


def test_suggested_params_match_proposed_solution() -> None:
    # The optimizer is updated via ``tell()`` using the vectors stored in ``user_attrs``,
    # so those vectors must correspond to the values actually evaluated.
    sampler = CatCmawmSampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=5)

    float_names = [n for n, d in SEARCH_SPACE.items() if isinstance(d, FloatDistribution)]
    int_names = [n for n, d in SEARCH_SPACE.items() if isinstance(d, IntDistribution)]

    for trial in study.trials:
        if "x" not in trial.user_attrs:  # First trial, before the search space is known.
            continue
        for proposed, name in zip(trial.user_attrs["x"], sorted(float_names)):
            assert proposed == pytest.approx(trial.params[name])
        for proposed, name in zip(trial.user_attrs["z"], sorted(int_names)):
            assert proposed == trial.params[name]


@pytest.mark.parametrize("distribution_types", ["f", "i", "c", "fi", "fc", "ic", "fic"])
def test_mixed_search_spaces(distribution_types: str) -> None:
    # ``solution.x`` and ``solution.z`` are empty when the corresponding parameter type
    # is absent, which is handled by separate guarded branches.
    def objective(trial: optuna.Trial) -> float:
        value = 0.0
        if "f" in distribution_types:
            value += trial.suggest_float("f1", -5, 5) ** 2
            value += trial.suggest_float("f2", -5, 5) ** 2
        if "i" in distribution_types:
            value += trial.suggest_int("i1", -3, 3) ** 2
            value += trial.suggest_int("i2", -3, 3) ** 2
        if "c" in distribution_types:
            value += float(trial.suggest_categorical("c1", [0, 1, 2]))
            value += float(trial.suggest_categorical("c2", [0, 1, 2]))
        return value

    study = optuna.create_study(sampler=CatCmawmSampler(seed=0))
    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
