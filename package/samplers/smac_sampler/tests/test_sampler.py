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
and modified to test SMACSampler.

Note that the following tests were omitted for now:
- test_reproducible_in_other_process
- test_reproducible
- test_nan_objective_value
- test_dynamic_range_objective
- test_conditional_parameter_objective

"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
import warnings

from _pytest.mark.structures import MarkDecorator
import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState
import optunahub
import pytest


# NOTE: This file content is mostly copied from the Optuna repository.
SMACSampler_ = optunahub.load_local_module(
    package="samplers/smac_sampler", registry_root="../../"
).SMACSampler


def SMACSampler(search_space: dict, seed: int | None = None):  # type: ignore
    return SMACSampler_(search_space=search_space, seed=seed, output_directory="/tmp/smac_output")


parametrize_sampler = pytest.mark.parametrize("sampler_class", [SMACSampler])
parametrize_relative_sampler = pytest.mark.parametrize("relative_sampler_class", [SMACSampler])
parametrize_multi_objective_sampler = pytest.mark.parametrize(
    "multi_objective_sampler_class", [SMACSampler]
)


sampler_class_with_seed: dict[str, Callable[[dict, int], BaseSampler]] = {
    "SMACSampler": lambda search_space, seed: SMACSampler(search_space, seed=seed)
}
param_sampler_with_seed = []
param_sampler_name_with_seed = []
for sampler_name, sampler_class in sampler_class_with_seed.items():
    param_sampler_with_seed.append(pytest.param(sampler_class, id=sampler_name))
    param_sampler_name_with_seed.append(pytest.param(sampler_name))
parametrize_sampler_with_seed = pytest.mark.parametrize("sampler_class", param_sampler_with_seed)
parametrize_sampler_name_with_seed = pytest.mark.parametrize(
    "sampler_name", param_sampler_name_with_seed
)


def parametrize_suggest_method(name: str) -> MarkDecorator:
    return pytest.mark.parametrize(
        f"suggest_method_{name}",
        [
            lambda t: t.suggest_float(name, 0, 10),
            lambda t: t.suggest_int(name, 0, 10),
            lambda t: t.suggest_categorical(name, [0, 1, 2]),
            lambda t: t.suggest_float(name, 0, 10, step=0.5),
            lambda t: t.suggest_float(name, 1e-7, 10, log=True),
            lambda t: t.suggest_int(name, 1, 10, log=True),
        ],
    )


@parametrize_sampler
@pytest.mark.parametrize(
    "distribution",
    [
        FloatDistribution(-1.0, 1.0),
        FloatDistribution(0.0, 1.0),
        FloatDistribution(-1.0, 0.0),
        FloatDistribution(1e-7, 1.0, log=True),
        FloatDistribution(-10, 10, step=0.1),
        FloatDistribution(-10.2, 10.2, step=0.1),
    ],
)
def test_float(
    sampler_class: Callable[[dict], BaseSampler], distribution: FloatDistribution
) -> None:
    study = optuna.study.create_study(sampler=sampler_class({"x": distribution}))
    points = np.array(
        [
            study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution)
            for _ in range(100)
        ]
    )
    assert np.all(points >= distribution.low)
    assert np.all(points <= distribution.high)
    assert not isinstance(
        study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution),
        np.floating,
    )

    if distribution.step is not None:
        # Check all points are multiples of distribution.step.
        points -= distribution.low
        points /= distribution.step
        round_points = np.round(points)
        np.testing.assert_almost_equal(round_points, points)


@parametrize_sampler
@pytest.mark.parametrize(
    "distribution",
    [
        IntDistribution(-10, 10),
        IntDistribution(0, 10),
        IntDistribution(-10, 0),
        IntDistribution(-10, 10, step=2),
        IntDistribution(0, 10, step=2),
        IntDistribution(-10, 0, step=2),
        IntDistribution(1, 100, log=True),
    ],
)
def test_int(sampler_class: Callable[[dict], BaseSampler], distribution: IntDistribution) -> None:
    study = optuna.study.create_study(sampler=sampler_class({"x": distribution}))
    points = np.array(
        [
            study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution)
            for _ in range(100)
        ]
    )
    assert np.all(points >= distribution.low)
    assert np.all(points <= distribution.high)
    assert not isinstance(
        study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution),
        np.integer,
    )


@parametrize_sampler
@pytest.mark.parametrize("choices", [(1, 2, 3), ("a", "b", "c"), (1, "a")])
def test_categorical(
    sampler_class: Callable[[dict], BaseSampler], choices: Sequence[CategoricalChoiceType]
) -> None:
    distribution = CategoricalDistribution(choices)

    study = optuna.study.create_study(sampler=sampler_class({"x": distribution}))

    def sample() -> float:
        trial = _create_new_trial(study)
        param_value = study.sampler.sample_independent(study, trial, "x", distribution)
        return float(distribution.to_internal_repr(param_value))

    points = np.asarray([sample() for i in range(100)])

    # 'x' value is corresponding to an index of distribution.choices.
    assert np.all(points >= 0)
    assert np.all(points <= len(distribution.choices) - 1)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)


@parametrize_relative_sampler
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
    relative_sampler_class: Callable[[dict], BaseSampler],
    x_distribution: BaseDistribution,
    y_distribution: BaseDistribution,
) -> None:
    search_space: dict[str, BaseDistribution] = dict(x=x_distribution, y=y_distribution)
    study = optuna.study.create_study(
        sampler=relative_sampler_class({"x": x_distribution, "y": y_distribution})
    )
    trial = study.ask(search_space)
    study.tell(trial, sum(trial.params.values()))

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


@parametrize_relative_sampler
def test_sample_relative_categorical(
    relative_sampler_class: Callable[[dict], BaseSampler],
) -> None:
    search_space: dict[str, BaseDistribution] = dict(
        x=CategoricalDistribution([1, 10, 100]), y=CategoricalDistribution([-1, -10, -100])
    )
    study = optuna.study.create_study(sampler=relative_sampler_class(search_space))
    trial = study.ask(search_space)
    study.tell(trial, sum(trial.params.values()))

    def sample() -> list[float]:
        params = study.sampler.sample_relative(study, _create_new_trial(study), search_space)
        return [params[name] for name in search_space]

    points = np.array([sample() for _ in range(7)])
    for i, distribution in enumerate(search_space.values()):
        assert isinstance(distribution, CategoricalDistribution)
        assert np.all([v in distribution.choices for v in points[:, i]])
    for param_value in sample():
        assert not isinstance(param_value, np.floating)
        assert not isinstance(param_value, np.integer)
        assert isinstance(param_value, int)


@parametrize_relative_sampler
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
    relative_sampler_class: Callable[[dict], BaseSampler], x_distribution: BaseDistribution
) -> None:
    search_space: dict[str, BaseDistribution] = dict(
        x=x_distribution, y=CategoricalDistribution([-1, -10, -100])
    )
    study = optuna.study.create_study(sampler=relative_sampler_class(search_space))
    trial = study.ask(search_space)
    study.tell(trial, sum(trial.params.values()))

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


@parametrize_sampler
def test_conditional_sample_independent(sampler_class: Callable[[dict], BaseSampler]) -> None:
    # This test case reproduces the error reported in #2734.
    # See https://github.com/optuna/optuna/pull/2734#issuecomment-857649769.

    categorical_distribution = CategoricalDistribution(choices=["x", "y"])
    dependent_distribution = CategoricalDistribution(choices=["a", "b"])
    study = optuna.study.create_study(
        sampler=sampler_class(
            {
                "category": categorical_distribution,
                "x": dependent_distribution,
                "y": dependent_distribution,
            }
        )
    )

    study.add_trial(
        optuna.create_trial(
            params={"category": "x", "x": "a"},
            distributions={"category": categorical_distribution, "x": dependent_distribution},
            value=0.1,
        )
    )

    study.add_trial(
        optuna.create_trial(
            params={"category": "y", "y": "b"},
            distributions={"category": categorical_distribution, "y": dependent_distribution},
            value=0.1,
        )
    )

    _trial = _create_new_trial(study)
    category = study.sampler.sample_independent(
        study, _trial, "category", categorical_distribution
    )
    assert category in ["x", "y"]
    value = study.sampler.sample_independent(study, _trial, category, dependent_distribution)
    assert value in ["a", "b"]


def _create_new_trial(study: Study) -> FrozenTrial:
    trial_id = study._storage.create_new_trial(study._study_id)
    return study._storage.get_trial(trial_id)


@parametrize_sampler
@parametrize_suggest_method("x")
def test_single_parameter_objective(
    sampler_class: Callable[[dict], BaseSampler], suggest_method_x: Callable[[Trial], float]
) -> None:
    def objective(trial: Trial) -> float:
        return suggest_method_x(trial)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        t = optuna.create_study().ask()
        objective(t)
        sampler = sampler_class(t.distributions)

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 3
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


@parametrize_sampler
@parametrize_suggest_method("x")
@parametrize_suggest_method("y")
def test_combination_of_different_distributions_objective(
    sampler_class: Callable[[dict], BaseSampler],
    suggest_method_x: Callable[[Trial], float],
    suggest_method_y: Callable[[Trial], float],
) -> None:
    def objective(trial: Trial) -> float:
        return suggest_method_x(trial) + suggest_method_y(trial)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        t = optuna.create_study().ask()
        objective(t)
        sampler = sampler_class(t.distributions)

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 3
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


@parametrize_multi_objective_sampler
@pytest.mark.parametrize(
    "distribution",
    [
        FloatDistribution(-1.0, 1.0),
        FloatDistribution(0.0, 1.0),
        FloatDistribution(-1.0, 0.0),
        FloatDistribution(1e-7, 1.0, log=True),
        FloatDistribution(-10, 10, step=0.1),
        FloatDistribution(-10.2, 10.2, step=0.1),
        IntDistribution(-10, 10),
        IntDistribution(0, 10),
        IntDistribution(-10, 0),
        IntDistribution(-10, 10, step=2),
        IntDistribution(0, 10, step=2),
        IntDistribution(-10, 0, step=2),
        IntDistribution(1, 100, log=True),
        CategoricalDistribution((1, 2, 3)),
        CategoricalDistribution(("a", "b", "c")),
        CategoricalDistribution((1, "a")),
    ],
)
def test_multi_objective_sample_independent(
    multi_objective_sampler_class: Callable[[dict], BaseSampler], distribution: BaseDistribution
) -> None:
    study = optuna.study.create_study(
        directions=["minimize", "maximize"],
        sampler=multi_objective_sampler_class({"x": distribution}),
    )
    for i in range(100):
        value = study.sampler.sample_independent(
            study, _create_new_trial(study), "x", distribution
        )
        assert distribution._contains(distribution.to_internal_repr(value))

        if not isinstance(distribution, CategoricalDistribution):
            # Please see https://github.com/optuna/optuna/pull/393 why this assertion is needed.
            assert not isinstance(value, np.floating)

        if isinstance(distribution, FloatDistribution):
            if distribution.step is not None:
                # Check the value is a multiple of `distribution.step` which is
                # the quantization interval of the distribution.
                value -= distribution.low
                value /= distribution.step
                round_value = np.round(value)
                np.testing.assert_almost_equal(round_value, value)


# We add tests for constant objective functions to ensure the reproducibility of sorting.
@parametrize_sampler_with_seed
@pytest.mark.slow
@pytest.mark.parametrize("objective_func", [lambda *args: sum(args), lambda *args: 0.0])
def test_reproducible(
    sampler_class: Callable[[dict, int], BaseSampler], objective_func: Any
) -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_float("a", 1, 9)
        b = trial.suggest_float("b", 1, 9, log=True)
        c = trial.suggest_float("c", 1, 9, step=1)
        d = trial.suggest_int("d", 1, 9)
        e = trial.suggest_int("e", 1, 9, log=True)
        f = trial.suggest_int("f", 1, 9, step=2)
        g = trial.suggest_categorical("g", range(1, 10))
        return objective_func(a, b, c, d, e, f, g)

    search_space = {
        "a": FloatDistribution(1, 9),
        "b": FloatDistribution(1, 9, log=True),
        "c": FloatDistribution(1, 9, step=1),
        "d": IntDistribution(1, 9),
        "e": IntDistribution(1, 9, log=True),
        "f": IntDistribution(1, 9, step=2),
        "g": CategoricalDistribution(range(1, 10)),
    }

    study = optuna.create_study(sampler=sampler_class(search_space, 1))
    study.optimize(objective, n_trials=15)

    study_same_seed = optuna.create_study(sampler=sampler_class(search_space, 1))
    study_same_seed.optimize(objective, n_trials=15)
    for i in range(15):
        assert study.trials[i].params == study_same_seed.trials[i].params

    study_different_seed = optuna.create_study(sampler=sampler_class(search_space, 2))
    study_different_seed.optimize(objective, n_trials=15)
    assert any(
        [study.trials[i].params != study_different_seed.trials[i].params for i in range(15)]
    )
