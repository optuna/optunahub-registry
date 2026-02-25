"""MIT License

Copyright (c) 2018 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

This file is taken from Optuna (https://github.com/optuna/optuna/blob/master/tests/samplers_tests/test_samplers.py)
and modified to test AutoSampler.
"""

from __future__ import annotations

from collections.abc import Callable
import pickle
from unittest.mock import patch

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.testing.pytest_samplers import BasicSamplerTestCase
from optuna.testing.pytest_samplers import MultiObjectiveSamplerTestCase
from optuna.testing.pytest_samplers import RelativeSamplerTestCase
from optuna.trial import FrozenTrial
from optuna.trial import Trial
import optunahub
import pytest


AutoSampler = optunahub.load_local_module(
    package="samplers/auto_sampler", registry_root="package/"
).AutoSampler


def _create_new_trial(study: Study) -> FrozenTrial:
    trial_id = study._storage.create_new_trial(study._study_id)
    return study._storage.get_trial(trial_id)


def _choose_sampler_in_auto_sampler_and_set_n_startup_trials_to_zero(study: optuna.Study) -> None:
    # NOTE(nabenabe): Choose a sampler inside AutoSampler.
    study.sampler.before_trial(study, trial=_create_new_trial(study))
    study.sampler._sampler._n_startup_trials = 0


# Test cases from Optuna's test suite


class TestSampler(BasicSamplerTestCase, MultiObjectiveSamplerTestCase, RelativeSamplerTestCase):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return AutoSampler

    # RelativeSamplerTestCase requires workarounds to test AutoSampler, so we override them here.
    # We explicitly inherit RelativeSamplerTestCase to ensure that all test cases in it are covered.
    @pytest.mark.parametrize(
        "x_distribution",
        [
            FloatDistribution(-1.0, 1.0),
            FloatDistribution(1e-7, 1.0, log=True),
            FloatDistribution(-10, 10, step=0.5),
            IntDistribution(3, 10),
            IntDistribution(1, 100, log=True),
            IntDistribution(3, 9, step=2),
        ],
    )
    @pytest.mark.parametrize(
        "y_distribution",
        [
            FloatDistribution(-1.0, 1.0),
            FloatDistribution(1e-7, 1.0, log=True),
            FloatDistribution(-10, 10, step=0.5),
            IntDistribution(3, 10),
            IntDistribution(1, 100, log=True),
            IntDistribution(3, 9, step=2),
        ],
    )
    def test_sample_relative_numerical(
        self,
        sampler: Callable[[], BaseSampler],
        x_distribution: BaseDistribution,
        y_distribution: BaseDistribution,
    ) -> None:
        search_space: dict[str, BaseDistribution] = dict(x=x_distribution, y=y_distribution)
        study = optuna.study.create_study(sampler=sampler())
        trial = study.ask(search_space)
        study.tell(trial, sum(trial.params.values()))
        _choose_sampler_in_auto_sampler_and_set_n_startup_trials_to_zero(study)

        def sample() -> list[int | float]:
            params = study.sampler.sample_relative(study, _create_new_trial(study), search_space)
            return [params[name] for name in search_space]

        points = np.array([sample() for _ in range(10)])
        for i, distribution in enumerate(search_space.values()):
            assert isinstance(
                distribution,
                (
                    FloatDistribution,
                    IntDistribution,
                ),
            )
            assert np.all(points[:, i] >= distribution.low)
            assert np.all(points[:, i] <= distribution.high)
        for param_value, distribution in zip(sample(), search_space.values()):
            assert not isinstance(param_value, np.floating)
            assert not isinstance(param_value, np.integer)
            if isinstance(distribution, IntDistribution):
                assert isinstance(param_value, int)
            else:
                assert isinstance(param_value, float)

    def test_sample_relative_categorical(self, sampler: Callable[[], BaseSampler]) -> None:
        search_space: dict[str, BaseDistribution] = dict(
            x=CategoricalDistribution([1, 10, 100]), y=CategoricalDistribution([-1, -10, -100])
        )
        study = optuna.study.create_study(sampler=sampler())
        trial = study.ask(search_space)
        study.tell(trial, sum(trial.params.values()))
        _choose_sampler_in_auto_sampler_and_set_n_startup_trials_to_zero(study)

        def sample() -> list[float]:
            params = study.sampler.sample_relative(study, _create_new_trial(study), search_space)
            return [params[name] for name in search_space]

        points = np.array([sample() for _ in range(10)])
        for i, distribution in enumerate(search_space.values()):
            assert isinstance(distribution, CategoricalDistribution)
            assert np.all([v in distribution.choices for v in points[:, i]])
        for param_value in sample():
            assert not isinstance(param_value, np.floating)
            assert not isinstance(param_value, np.integer)
            assert isinstance(param_value, int)

    @pytest.mark.parametrize(
        "x_distribution",
        [
            FloatDistribution(-1.0, 1.0),
            FloatDistribution(1e-7, 1.0, log=True),
            FloatDistribution(-10, 10, step=0.5),
            IntDistribution(1, 10),
            IntDistribution(1, 100, log=True),
        ],
    )
    def test_sample_relative_mixed(
        self, sampler: Callable[[], BaseSampler], x_distribution: BaseDistribution
    ) -> None:
        search_space: dict[str, BaseDistribution] = dict(
            x=x_distribution, y=CategoricalDistribution([-1, -10, -100])
        )
        study = optuna.study.create_study(sampler=sampler())
        trial = study.ask(search_space)
        study.tell(trial, sum(trial.params.values()))
        _choose_sampler_in_auto_sampler_and_set_n_startup_trials_to_zero(study)

        def sample() -> list[float]:
            params = study.sampler.sample_relative(study, _create_new_trial(study), search_space)
            return [params[name] for name in search_space]

        points = np.array([sample() for _ in range(10)])
        assert isinstance(
            search_space["x"],
            (
                FloatDistribution,
                IntDistribution,
            ),
        )
        assert np.all(points[:, 0] >= search_space["x"].low)
        assert np.all(points[:, 0] <= search_space["x"].high)
        assert isinstance(search_space["y"], CategoricalDistribution)
        assert np.all([v in search_space["y"].choices for v in points[:, 1]])
        for param_value, distribution in zip(sample(), search_space.values()):
            assert not isinstance(param_value, np.floating)
            assert not isinstance(param_value, np.integer)
            if isinstance(
                distribution,
                (
                    IntDistribution,
                    CategoricalDistribution,
                ),
            ):
                assert isinstance(param_value, int)
            else:
                assert isinstance(param_value, float)

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_cache_is_invalidated(
        self,
        sampler: Callable[[], BaseSampler],
        n_jobs: int,
    ) -> None:
        sampler_ = sampler()
        original_before_trial = sampler_.before_trial

        def mock_before_trial(study: Study, trial: FrozenTrial) -> None:
            assert study._thread_local.cached_all_trials is None
            original_before_trial(study, trial)

        with patch.object(sampler_, "before_trial", side_effect=mock_before_trial):
            study = optuna.study.create_study(sampler=sampler_)

            def objective(trial: Trial) -> float:
                assert trial._relative_params is None

                trial.suggest_float("x", -10, 10)
                trial.suggest_float("y", -10, 10)
                assert trial._relative_params is not None
                return -1

            study.optimize(objective, n_trials=10, n_jobs=n_jobs)


# AutoSampler-specific tests

parametrize_constraints = pytest.mark.parametrize("use_constraint", [True, False])


def objective_1d(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    return x**2


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    return x**2 + y**2


def objective_with_categorical(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    c = trial.suggest_categorical("c", [True, False])
    return x**2 + y**2 if c else (x - 2) ** 2 + (y - 2) ** 2


def multi_objective_with_categorical(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    _ = trial.suggest_categorical("c", [True, False])
    return x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2


def many_objective_with_categorical(trial: optuna.Trial) -> tuple[float, float, float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    _ = trial.suggest_categorical("c", [True, False])
    return x**2 + y**2, x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2, (x + 2) ** 2 + (y + 2) ** 2


def objective_with_conditional(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    if trial.number < 15:
        y = trial.suggest_int("y", -5, 5)
        return x**2 + y**2
    else:
        z = trial.suggest_float("z", -5, 5)
        return x**2 + z**2


def multi_objective_with_conditional(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    if trial.number < 15:
        y = trial.suggest_int("y", -5, 5)
        return x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2
    else:
        z = trial.suggest_float("z", -5, 5)
        return x**2 + z**2, (x - 2) ** 2 + (z - 2) ** 2


def many_objective_with_conditional(trial: optuna.Trial) -> tuple[float, float, float, float]:
    x = trial.suggest_float("x", -5, 5)
    if trial.number < 15:
        y = trial.suggest_int("y", -5, 5)
        return x**2 + y**2, x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2, (x + 2) ** 2 + (y + 2) ** 2
    else:
        z = trial.suggest_float("z", -5, 5)
        return x**2 + z**2, x**2 + z**2, (x - 2) ** 2 + (z - 2) ** 2, (x + 2) ** 2 + (z + 2) ** 2


def multi_objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    return x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2


def many_objective(trial: optuna.Trial) -> tuple[float, float, float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    return x**2 + y**2, x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2, (x + 2) ** 2 + (y + 2) ** 2


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
def test_choose_for_multi_objective(use_constraint: bool) -> None:
    n_trials_of_nsgaii = 100
    n_trials_before_nsgaii = 15
    auto_sampler = AutoSampler(constraints_func=constraints_func if use_constraint else None)
    auto_sampler._MAX_BUDGET_FOR_MULTI_GP_AND_TPE = n_trials_before_nsgaii
    study = optuna.create_study(sampler=auto_sampler, directions=["minimize"] * 2)
    study.optimize(multi_objective, n_trials=n_trials_before_nsgaii + n_trials_of_nsgaii)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["GPSampler"] * (n_trials_before_nsgaii - 1) + [
        "NSGAIISampler"
    ] * n_trials_of_nsgaii == sampler_names
    if use_constraint:
        _check_constraints_of_all_trials(study)


@parametrize_constraints
def test_choose_for_many_objective(use_constraint: bool) -> None:
    n_trials_of_nsgaiii = 100
    n_trials_before_nsgaiii = 15
    auto_sampler = AutoSampler(constraints_func=constraints_func if use_constraint else None)
    auto_sampler._MAX_BUDGET_FOR_MULTI_GP_AND_TPE = n_trials_before_nsgaiii
    study = optuna.create_study(sampler=auto_sampler, directions=["minimize"] * 4)
    study.optimize(many_objective, n_trials=n_trials_before_nsgaiii + n_trials_of_nsgaiii)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["TPESampler"] * (n_trials_before_nsgaiii - 1) + [
        "NSGAIIISampler"
    ] * n_trials_of_nsgaiii == sampler_names
    if use_constraint:
        _check_constraints_of_all_trials(study)


def test_choose_cmaes() -> None:
    # This test must be performed with a numerical objective function.
    n_trials_of_cmaes = 100
    n_trials_before_cmaes = 20
    auto_sampler = AutoSampler()
    auto_sampler._MAX_BUDGET_FOR_SINGLE_GP = n_trials_before_cmaes
    study = optuna.create_study(sampler=auto_sampler)
    study.optimize(objective, n_trials=n_trials_of_cmaes + n_trials_before_cmaes)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["GPSampler"] * (n_trials_before_cmaes - 1) + [
        "CmaEsSampler"
    ] * n_trials_of_cmaes == sampler_names


def test_choose_cmaes_for_1d() -> None:
    # This test must be performed with a numerical objective function.
    # For 1d problems, TPESampler will be chosen instead of CmaEsSampler.
    n_trials_of_cmaes = 100
    n_trials_before_cmaes = 20
    auto_sampler = AutoSampler()
    auto_sampler._MAX_BUDGET_FOR_SINGLE_GP = n_trials_before_cmaes
    study = optuna.create_study(sampler=auto_sampler)
    study.optimize(objective_1d, n_trials=n_trials_of_cmaes + n_trials_before_cmaes)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["GPSampler"] * (n_trials_before_cmaes - 1) + [
        "CmaEsSampler"
    ] * n_trials_of_cmaes == sampler_names


def test_choose_for_single_objective_with_constraints() -> None:
    n_trials_of_tpe = 100
    n_trials_before_tpe = 20
    auto_sampler = AutoSampler(constraints_func=constraints_func)
    auto_sampler._MAX_BUDGET_FOR_SINGLE_GP = n_trials_before_tpe
    study = optuna.create_study(sampler=auto_sampler)
    study.optimize(objective, n_trials=n_trials_before_tpe + n_trials_of_tpe)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["GPSampler"] * (n_trials_before_tpe - 1) + [
        "TPESampler"
    ] * n_trials_of_tpe == sampler_names


def test_choose_tpe_with_categorical_params() -> None:
    n_trials = 30
    auto_sampler = AutoSampler()
    study = optuna.create_study(sampler=auto_sampler)
    study.optimize(objective_with_categorical, n_trials=n_trials)
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["TPESampler"] * (n_trials - 1) == sampler_names


def test_choose_for_multi_objective_with_categorical() -> None:
    n_trials_before_nsgaii = 15
    n_trials_of_nsgaii = 100
    auto_sampler = AutoSampler()
    auto_sampler._MAX_BUDGET_FOR_MULTI_GP_AND_TPE = n_trials_before_nsgaii
    study = optuna.create_study(sampler=auto_sampler, directions=["minimize"] * 2)
    study.optimize(
        multi_objective_with_categorical, n_trials=n_trials_before_nsgaii + n_trials_of_nsgaii
    )
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["TPESampler"] * (n_trials_before_nsgaii - 1) + [
        "NSGAIISampler"
    ] * n_trials_of_nsgaii == sampler_names


def test_choose_for_many_objective_with_categorical() -> None:
    n_trials_before_nsgaiii = 15
    n_trials_of_nsgaiii = 100
    auto_sampler = AutoSampler()
    auto_sampler._MAX_BUDGET_FOR_MULTI_GP_AND_TPE = n_trials_before_nsgaiii
    study = optuna.create_study(sampler=auto_sampler, directions=["minimize"] * 4)
    study.optimize(
        many_objective_with_categorical, n_trials=n_trials_before_nsgaiii + n_trials_of_nsgaiii
    )
    sampler_names = _get_used_sampler_names(study)
    assert ["RandomSampler"] + ["TPESampler"] * (n_trials_before_nsgaiii - 1) + [
        "NSGAIIISampler"
    ] * n_trials_of_nsgaiii == sampler_names


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


def test_choose_for_multi_objective_with_conditional() -> None:
    n_trials = 30
    auto_sampler = AutoSampler()
    study = optuna.create_study(sampler=auto_sampler, directions=["minimize"] * 2)
    study.optimize(multi_objective_with_conditional, n_trials=n_trials)
    sampler_names = _get_used_sampler_names(study)
    # NOTE(nabenabe): When the conditional parameter is detected for the first time, GPSampler
    # simply falls back to sample_independent, so sample_relative of GPSampler is called 15 times.
    print(sampler_names)
    assert ["RandomSampler"] + ["GPSampler"] * 15 + ["TPESampler"] * (
        n_trials - 16
    ) == sampler_names


def test_choose_for_many_objective_with_conditional() -> None:
    n_trials = 30
    auto_sampler = AutoSampler()
    study = optuna.create_study(sampler=auto_sampler, directions=["minimize"] * 4)
    study.optimize(many_objective_with_conditional, n_trials=n_trials)
    sampler_names = _get_used_sampler_names(study)
    # NOTE(nabenabe): When the conditional parameter is detected for the first time, GPSampler
    # simply falls back to sample_independent, so sample_relative of GPSampler is called 15 times.
    print(sampler_names)
    assert ["RandomSampler"] + ["TPESampler"] * (n_trials - 1) == sampler_names


def test_multi_thread() -> None:
    n_trials = 30
    auto_sampler = AutoSampler()
    auto_sampler._MAX_BUDGET_FOR_SINGLE_GP = 10
    study = optuna.create_study(sampler=auto_sampler)
    study.optimize(objective, n_trials=n_trials)
    sampler_names = _get_used_sampler_names(study)
    assert "RandomSampler" in sampler_names
    assert "GPSampler" in sampler_names
    assert "CmaEsSampler" in sampler_names


def test_picklize() -> None:
    pickle.loads(pickle.dumps(AutoSampler()))
