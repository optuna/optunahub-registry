from __future__ import annotations

from collections.abc import Sequence
import warnings

import optuna
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.trial import FrozenTrial
import optunahub
import pytest


# NOTE(nabenabe): This file content is mostly copied from the Optuna repository.
CARBOSampler = optunahub.load_local_module(
    package="samplers/carbo", registry_root="package/"
).CARBOSampler


@pytest.mark.parametrize("constraint_value", [-1.0, 0.0, 1.0, -float("inf"), float("inf")])
def test_constraints_func(constraint_value: float) -> None:
    n_trials = 5
    constraints_func_call_count = 0

    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        nonlocal constraints_func_call_count
        constraints_func_call_count += 1

        return (constraint_value + trial.number,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = CARBOSampler(
            n_startup_trials=2, constraints_func=constraints_func, n_local_search=4
        )

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=n_trials)

    assert len(study.trials) == n_trials
    assert constraints_func_call_count == n_trials
    for trial in study.trials:
        for x, y in zip(trial.system_attrs[_CONSTRAINTS_KEY], (constraint_value + trial.number,)):
            assert x == y


def test_constraints_func_nan() -> None:
    n_trials = 5

    def constraints_func(_: FrozenTrial) -> Sequence[float]:
        return (float("nan"),)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = CARBOSampler(
            n_startup_trials=2, constraints_func=constraints_func, n_local_search=4
        )

    study = optuna.create_study(direction="minimize", sampler=sampler)
    with pytest.raises(ValueError):
        study.optimize(
            lambda t: t.suggest_float("x", 0, 1),
            n_trials=n_trials,
        )

    trials = study.get_trials()
    assert len(trials) == 1  # The error stops optimization, but completed trials are recorded.
    assert all(0 <= x <= 1 for x in trials[0].params.values())  # The params are normal.
    assert trials[0].values == list(trials[0].params.values())  # The values are normal.
    assert trials[0].system_attrs[_CONSTRAINTS_KEY] is None  # None is set for constraints.
