from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState
import optunahub
import pytest


MetaLearnTPESampler = optunahub.load_local_module(
    package="samplers/meta_learn_tpe", registry_root="package/"
).MetaLearnTPESampler


def _add_completed_trials(
    study: optuna.Study, *, large_params: bool = False, n_trials: int = 10
) -> None:
    for i in range(n_trials):
        x = i / 10
        y = 1 - x
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": x, "y": ("a" if i % 2 else "b") * 3000 if large_params else y},
                distributions={
                    "x": FloatDistribution(0, 1),
                    "y": CategoricalDistribution(["a" * 3000, "b" * 3000])
                    if large_params
                    else FloatDistribution(0, 1),
                },
                values=[x**2 + y**2] if len(study.directions) == 1 else [x**2, y**2],
            )
        )


@pytest.mark.parametrize("constant_liar", [False, True])
@pytest.mark.parametrize("n_objectives", [1, 2])
@pytest.mark.parametrize("storage_type", ["memory", "sqlite"])
def test_pending_parameters_reach_other_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant_liar: bool,
    n_objectives: int,
    storage_type: str,
) -> None:
    _check_pending_parameters(tmp_path, monkeypatch, constant_liar, n_objectives, storage_type)


def test_large_pending_parameters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The full relative proposal needs multiple storage entries.
    _check_pending_parameters(tmp_path, monkeypatch, True, 1, "sqlite", large_params=True)


def _check_pending_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant_liar: bool,
    n_objectives: int,
    storage_type: str,
    large_params: bool = False,
) -> None:
    directions = ["minimize"] * n_objectives
    source = optuna.create_study(directions=directions)
    _add_completed_trials(source, large_params=large_params)
    sampler = MetaLearnTPESampler(
        source_studies=[source], n_startup_trials=5, seed=0, constant_liar=constant_liar
    )
    storage: Any = (
        f"sqlite:///{tmp_path / 'study.db'}"
        if storage_type == "sqlite"
        else optuna.storages.InMemoryStorage()
    )
    study = optuna.create_study(directions=directions, storage=storage, sampler=sampler)
    _add_completed_trials(study, large_params=large_params)
    pending = study.ask()
    pending.suggest_float("x", 0, 1)
    assert "y" in pending.relative_params
    assert "y" not in study.trials[-1].params

    worker_sampler = MetaLearnTPESampler(
        source_studies=[source], n_startup_trials=5, seed=1, constant_liar=constant_liar
    )
    worker = optuna.load_study(
        study_name=study.study_name, storage=storage, sampler=worker_sampler
    )
    captured: dict[bool, list[optuna.trial.FrozenTrial]] = {}
    build = worker_sampler._target_sampler._build_parzen_estimator

    def capture(
        study: optuna.Study,
        search_space: dict[str, Any],
        trials: list[optuna.trial.FrozenTrial],
        handle_below: bool,
    ) -> Any:
        captured[handle_below] = trials
        return build(study, search_space, trials, handle_below)

    monkeypatch.setattr(worker_sampler._target_sampler, "_build_parzen_estimator", capture)
    current = worker.ask()
    current.suggest_float("x", 0, 1)
    above = {trial.number: trial for trial in captured[False]}
    assert (pending.number in above) == constant_liar
    assert all(trial.number != current.number for trials in captured.values() for trial in trials)
    # Ten finished trials put one trial below, even with a pending trial present.
    assert len(captured[True]) == 1
    if constant_liar:
        assert above[pending.number].params == pending.relative_params
    # Reading the proposal must not persist the not-yet-suggested parameter.
    assert "y" not in study.trials[pending.number].params
    assert "y" not in study.trials[pending.number].distributions


def test_running_trials_do_not_end_startup() -> None:
    source = optuna.create_study()
    _add_completed_trials(source)
    sampler = MetaLearnTPESampler(
        source_studies=[source], n_startup_trials=5, seed=0, constant_liar=True
    )
    study = optuna.create_study(sampler=sampler)
    _add_completed_trials(study, n_trials=4)
    for _ in range(10):
        trial = study.ask()
        trial.suggest_float("x", 0, 1)
        trial.suggest_float("y", 0, 1)
        assert trial.relative_params == {}


def test_current_trial_is_not_penalized(monkeypatch: pytest.MonkeyPatch) -> None:
    source = optuna.create_study()
    _add_completed_trials(source)
    sampler = MetaLearnTPESampler(
        source_studies=[source], n_startup_trials=5, seed=0, constant_liar=True
    )
    study = optuna.create_study(sampler=sampler)
    _add_completed_trials(study)
    current = study.ask()
    current.suggest_float("x", 0, 1)
    current.suggest_float("y", 0, 1)
    build = sampler._target_sampler._build_parzen_estimator

    def capture(
        study: optuna.Study,
        search_space: dict[str, Any],
        trials: list[optuna.trial.FrozenTrial],
        handle_below: bool,
    ) -> Any:
        assert all(trial.number != current.number for trial in trials)
        return build(study, search_space, trials, handle_below)

    monkeypatch.setattr(sampler._target_sampler, "_build_parzen_estimator", capture)
    frozen = study.trials[-1]
    assert frozen.state == TrialState.RUNNING
    params = sampler.sample_relative(study, frozen, frozen.distributions)
    assert set(params) == {"x", "y"}
