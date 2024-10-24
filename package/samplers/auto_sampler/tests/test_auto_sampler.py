from __future__ import annotations

import optuna
import optunahub
import pytest


# TODO(nabaenabe): Add the CI for this sampler.
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


AutoSampler = optunahub.load_local_module(
    package="samplers/auto_sampler", registry_root="package/"
).AutoSampler

parametrize_constraints = pytest.mark.parametrize("use_constraint", [True, False])


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    return x**2 + y**2


def objective_with_categorical(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    c = trial.suggest_categorical("c", [True, False])
    return x**2 + y**2 if c else (x - 2) ** 2 + (y - 2) ** 2


def objective_with_conditional(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    if trial.number < 15:
        y = trial.suggest_int("y", -5, 5)
        return x**2 + y**2
    else:
        z = trial.suggest_float("z", -5, 5)
        return x**2 + z**2


def multi_objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    return x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2


def many_objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    return (x**2 + y**2, x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2, (x + 2) ** 2 + (y + 2) ** 2)


def constraints_func(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    return (float(trial.params["x"] >= 2),)


def _get_used_sampler_names(study: optuna.Study) -> list[str]:
    return [
        study._storage.get_trial_system_attrs(t._trial_id).get("auto:sampler")
        for t in study.trials
    ]


def _check_constraints_of_all_trials(study: optuna.Study) -> None:
    target_key = optuna.samplers._base._CONSTRAINTS_KEY
    assert all(
        target_key in study._storage.get_trial_system_attrs(t._trial_id) for t in study.trials
    )


@parametrize_constraints
def test_choose_nsga3(use_constraint: bool) -> None:
    n_trials_of_nsga = 100
    n_trials_before_nsga = 100
    auto_sampler = AutoSampler(constraints_func=constraints_func if use_constraint else None)
    auto_sampler._N_COMPLETE_TRIALS_FOR_NSGA = n_trials_before_nsga
    study = optuna.create_study(sampler=auto_sampler, directions=["minimize"] * 4)
    study.optimize(many_objective, n_trials=n_trials_before_nsga + n_trials_of_nsga)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["TPESampler"] * (n_trials_before_nsga - 1) + [
        "NSGAIIISampler"
    ] * n_trials_of_nsga == sampler_names
    if use_constraint:
        _check_constraints_of_all_trials(study)


@parametrize_constraints
def test_choose_nsga2(use_constraint: bool) -> None:
    n_trials_of_nsga = 100
    n_trials_before_nsga = 100
    auto_sampler = AutoSampler(constraints_func=constraints_func if use_constraint else None)
    auto_sampler._N_COMPLETE_TRIALS_FOR_NSGA = n_trials_before_nsga
    study = optuna.create_study(sampler=auto_sampler, directions=["minimize"] * 2)
    study.optimize(multi_objective, n_trials=n_trials_before_nsga + n_trials_of_nsga)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["TPESampler"] * (n_trials_before_nsga - 1) + [
        "NSGAIISampler"
    ] * n_trials_of_nsga == sampler_names
    if use_constraint:
        _check_constraints_of_all_trials(study)


def test_choose_cmaes() -> None:
    # This test must be performed with a numerical objective function.
    n_trials_of_cmaes = 100
    n_trials_before_cmaes = 20
    auto_sampler = AutoSampler()
    auto_sampler._N_COMPLETE_TRIALS_FOR_CMAES = n_trials_before_cmaes
    study = optuna.create_study(sampler=auto_sampler)
    study.optimize(objective, n_trials=n_trials_of_cmaes + n_trials_before_cmaes)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["GPSampler"] * (n_trials_before_cmaes - 1) + [
        "CmaEsSampler"
    ] * n_trials_of_cmaes == sampler_names


def test_choose_tpe_in_single_with_constraints() -> None:
    n_trials = 30
    auto_sampler = AutoSampler(constraints_func=constraints_func)
    study = optuna.create_study(sampler=auto_sampler)
    study.optimize(objective, n_trials=n_trials)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["TPESampler"] * (n_trials - 1) == sampler_names


def test_choose_tpe_with_categorical_params() -> None:
    n_trials = 30
    auto_sampler = AutoSampler()
    study = optuna.create_study(sampler=auto_sampler)
    study.optimize(objective_with_categorical, n_trials=n_trials)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["TPESampler"] * (n_trials - 1) == sampler_names


def test_choose_tpe_with_conditional_params() -> None:
    n_trials = 30
    auto_sampler = AutoSampler()
    study = optuna.create_study(sampler=auto_sampler)
    study.optimize(objective_with_conditional, n_trials=n_trials)
    sampler_names = _get_used_sampler_names(study)
    # NOTE(nabenabe): When the conditional parameter is detected for the first time, GPSampler
    # simply falls back to sample_independent, so sample_relative of GPSampler is called 15 times.
    assert ["RandomSampler"] + ["GPSampler"] * 15 + ["TPESampler"] * (
        n_trials - 16
    ) == sampler_names


# TODO: Add a test with enqueue_trial.
