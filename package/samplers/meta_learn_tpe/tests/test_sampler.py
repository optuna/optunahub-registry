from __future__ import annotations

from typing import Callable
from typing import Sequence

import optuna
from optuna.trial import TrialState
import optunahub
import pytest


MetaLearnTPESampler = optunahub.load_local_module(
    package="samplers/meta_learn_tpe", registry_root="package/"
).MetaLearnTPESampler


def _single_objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


def _multi_objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2


@pytest.mark.parametrize(
    "directions, objective",
    [
        (["minimize"], _single_objective),
        (["minimize", "minimize"], _multi_objective),
    ],
)
@pytest.mark.parametrize("constant_liar", [False, True])
def test_optimize_after_startup_trials(
    directions: Sequence[str],
    objective: Callable[[optuna.Trial], float | tuple[float, float]],
    constant_liar: bool,
) -> None:
    source = optuna.create_study(
        directions=directions, sampler=optuna.samplers.RandomSampler(seed=0)
    )
    source.optimize(objective, n_trials=30)

    sampler = MetaLearnTPESampler(
        source_studies=[source], n_startup_trials=5, seed=0, constant_liar=constant_liar
    )
    study = optuna.create_study(directions=directions, sampler=sampler)
    # The trials after ``n_startup_trials`` are sampled by the meta-learning TPE.
    study.optimize(objective, n_trials=15)

    assert len(study.trials) == 15
    assert all(trial.state == TrialState.COMPLETE for trial in study.trials)
