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
from collections.abc import Sequence
import multiprocessing
from multiprocessing.managers import DictProxy
import os
from typing import Any
from unittest.mock import patch
import warnings
import logging
from datetime import datetime


from _pytest.fixtures import SubRequest
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


# NOTE(nabenabe): This file content is mostly copied from the Optuna repository.
The_Sampler = optunahub.load_local_module(
    package="package/samplers/de", registry_root="/home/j/PycharmProjects/optunahub-registry"
).DESampler


parametrize_sampler = pytest.mark.parametrize("sampler_class", [The_Sampler])
parametrize_relative_sampler = pytest.mark.parametrize("relative_sampler_class", [The_Sampler])
parametrize_multi_objective_sampler = pytest.mark.parametrize(
    "multi_objective_sampler_class", [The_Sampler]
)


sampler_class_with_seed: dict[str, Callable[[int], BaseSampler]] = {
    "TheSampler": lambda seed: The_Sampler(seed=seed)
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


def _choose_sampler_in_auto_sampler_and_set_n_startup_trials_to_zero(study: optuna.Study) -> None:
    # NOTE(nabenabe): Choose a sampler inside AutoSampler.
    study.sampler.before_trial(study, trial=_create_new_trial(study))
    study.sampler._sampler._n_startup_trials = 0


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
def test_float(sampler_class: Callable[[], BaseSampler], distribution: FloatDistribution) -> None:
    study = optuna.study.create_study(sampler=sampler_class())
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
def test_int(sampler_class: Callable[[], BaseSampler], distribution: IntDistribution) -> None:
    study = optuna.study.create_study(sampler=sampler_class())
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
    sampler_class: Callable[[], BaseSampler], choices: Sequence[CategoricalChoiceType]
) -> None:
    distribution = CategoricalDistribution(choices)

    study = optuna.study.create_study(sampler=sampler_class())

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
    relative_sampler_class: Callable[[], BaseSampler],
    x_distribution: BaseDistribution,
    y_distribution: BaseDistribution,
) -> None:
    search_space: dict[str, BaseDistribution] = dict(x=x_distribution, y=y_distribution)
    study = optuna.study.create_study(sampler=relative_sampler_class())
    trial = study.ask(search_space)
    study.tell(trial, sum(trial.params.values()))

    """
    This test checks if the relative sampler samples parameters correctly. Specifically, it checks
    if the sampled parameters are within the search space and have the correct type. The test
    samples 10 points and checks if the sampled parameters are within the search space and have the
    correct type.
    """

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
        ), "The distribution must be either FloatDistribution or IntDistribution."
        assert np.all(points[:, i] >= distribution.low), "The sampled value must be >= low."
        assert np.all(points[:, i] <= distribution.high), "The sampled value must be <= high."
    for param_value, distribution in zip(sample(), search_space.values()):
        assert not isinstance(param_value, np.floating), f"The sampled value must not be a numpy float, instead it is {param_value}"
        assert not isinstance(param_value, np.integer), f"The sampled value must not be a numpy integer, instead it is {param_value}"
        if isinstance(distribution, IntDistribution):
            assert isinstance(param_value, int), "The sampled value must be an integer."
        else:
            assert isinstance(param_value, float), "The sampled value must be a float."


@parametrize_relative_sampler
def test_sample_relative_categorical(relative_sampler_class: Callable[[], BaseSampler]) -> None:
    search_space: dict[str, BaseDistribution] = dict(
        x=CategoricalDistribution([1, 10, 100]), y=CategoricalDistribution([-1, -10, -100])
    )
    study = optuna.study.create_study(sampler=relative_sampler_class())
    trial = study.ask(search_space)
    study.tell(trial, sum(trial.params.values()))

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
    relative_sampler_class: Callable[[], BaseSampler], x_distribution: BaseDistribution
) -> None:
    search_space: dict[str, BaseDistribution] = dict(
        x=x_distribution, y=CategoricalDistribution([-1, -10, -100])
    )
    study = optuna.study.create_study(sampler=relative_sampler_class())
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
def test_conditional_sample_independent(sampler_class: Callable[[], BaseSampler]) -> None:
    # This test case reproduces the error reported in #2734.
    # See https://github.com/optuna/optuna/pull/2734#issuecomment-857649769.

    study = optuna.study.create_study(sampler=sampler_class())
    categorical_distribution = CategoricalDistribution(choices=["x", "y"])
    dependent_distribution = CategoricalDistribution(choices=["a", "b"])

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
def test_nan_objective_value(sampler_class: Callable[[], BaseSampler]) -> None:
    study = optuna.create_study(sampler=sampler_class())

    def objective(trial: Trial, base_value: float) -> float:
        return trial.suggest_float("x", 0.1, 0.2) + base_value

    # Non NaN objective values.
    for i in range(10, 1, -1):
        study.optimize(lambda t: objective(t, i), n_trials=1, catch=())
    assert int(study.best_value) == 2

    # NaN objective values.
    study.optimize(lambda t: objective(t, float("nan")), n_trials=1, catch=())
    assert int(study.best_value) == 2

    # Non NaN objective value.
    study.optimize(lambda t: objective(t, 1), n_trials=1, catch=())
    assert int(study.best_value) == 1


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
    multi_objective_sampler_class: Callable[[], BaseSampler], distribution: BaseDistribution
) -> None:
    study = optuna.study.create_study(
        directions=["minimize", "maximize"], sampler=multi_objective_sampler_class()
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


@parametrize_sampler
def test_sample_single_distribution(sampler_class: Callable[[], BaseSampler]) -> None:
    relative_search_space = {
        "a": CategoricalDistribution([1]),
        "b": IntDistribution(low=1, high=1),
        "c": IntDistribution(low=1, high=1, log=True),
        "d": FloatDistribution(low=1.0, high=1.0),
        "e": FloatDistribution(low=1.0, high=1.0, log=True),
        "f": FloatDistribution(low=1.0, high=1.0, step=1.0),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()
    study = optuna.study.create_study(sampler=sampler)

    # We need to test the construction of the model, so we should set `n_trials >= 2`.
    for _ in range(2):
        trial = study.ask(fixed_distributions=relative_search_space)
        study.tell(trial, 1.0)
        for param_name in relative_search_space.keys():
            assert trial.params[param_name] == 1


@parametrize_sampler
@parametrize_suggest_method("x")
def test_single_parameter_objective(
    sampler_class: Callable[[], BaseSampler], suggest_method_x: Callable[[Trial], float]
) -> None:
    def objective(trial: Trial) -> float:
        return suggest_method_x(trial)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


@parametrize_sampler
def test_conditional_parameter_objective(sampler_class: Callable[[], BaseSampler]) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_categorical("x", [True, False])
        if x:
            return trial.suggest_float("y", 0, 1)
        return trial.suggest_float("z", 0, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


@parametrize_sampler
@parametrize_suggest_method("x")
@parametrize_suggest_method("y")
def test_combination_of_different_distributions_objective(
    sampler_class: Callable[[], BaseSampler],
    suggest_method_x: Callable[[Trial], float],
    suggest_method_y: Callable[[Trial], float],
) -> None:
    def objective(trial: Trial) -> float:
        return suggest_method_x(trial) + suggest_method_y(trial)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=3)

    assert len(study.trials) == 3
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


@parametrize_sampler
@pytest.mark.parametrize(
    "second_low,second_high",
    [
        (0, 5),  # Narrow range.
        (0, 20),  # Expand range.
        (20, 30),  # Set non-overlapping range.
    ],
)
def test_dynamic_range_objective(
    sampler_class: Callable[[], BaseSampler], second_low: int, second_high: int
) -> None:
    def objective(trial: Trial, low: int, high: int) -> float:
        v = trial.suggest_float("x", low, high)
        v += trial.suggest_int("y", low, high)
        return v

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(lambda t: objective(t, 0, 10), n_trials=10)
    study.optimize(lambda t: objective(t, second_low, second_high), n_trials=10)

    assert len(study.trials) == 20
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


# We add tests for constant objective functions to ensure the reproducibility of sorting.
@parametrize_sampler_with_seed
@pytest.mark.slow
@pytest.mark.parametrize("objective_func", [lambda *args: sum(args), lambda *args: 0.0])
def test_reproducible(sampler_class: Callable[[int], BaseSampler], objective_func: Any) -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_float("a", 1, 9)
        b = trial.suggest_float("b", 1, 9, log=True)
        c = trial.suggest_float("c", 1, 9, step=1)
        d = trial.suggest_int("d", 1, 9)
        e = trial.suggest_int("e", 1, 9, log=True)
        f = trial.suggest_int("f", 1, 9, step=2)
        g = trial.suggest_categorical("g", range(1, 10))
        return objective_func(a, b, c, d, e, f, g)

    """
    This test checks if the sampler is reproducible. Specifically, it checks if the same set of
    parameters are suggested for the same seed. The test optimizes a constant objective function
    with 15 trials and checks if the parameters are the same for the same seed.
    """

    study = optuna.create_study(sampler=sampler_class(1))
    study.optimize(objective, n_trials=15)

    study_same_seed = optuna.create_study(sampler=sampler_class(1))
    study_same_seed.optimize(objective, n_trials=15)
    for i in range(15):
        assert study.trials[i].params == study_same_seed.trials[i].params

    study_different_seed = optuna.create_study(sampler=sampler_class(2))
    study_different_seed.optimize(objective, n_trials=15)
    assert any(
        [study.trials[i].params != study_different_seed.trials[i].params for i in range(15)]
    )


@pytest.mark.slow
@parametrize_sampler_with_seed
def test_reseed_rng_change_sampling(sampler_class: Callable[[int], BaseSampler]) -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_float("a", 1, 9)
        b = trial.suggest_float("b", 1, 9, log=True)
        c = trial.suggest_float("c", 1, 9, step=1)
        d = trial.suggest_int("d", 1, 9)
        e = trial.suggest_int("e", 1, 9, log=True)
        f = trial.suggest_int("f", 1, 9, step=2)
        g = trial.suggest_categorical("g", range(1, 10))
        return a + b + c + d + e + f + g

    sampler = sampler_class(1)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=15)

    sampler_different_seed = sampler_class(1)
    sampler_different_seed.reseed_rng()
    study_different_seed = optuna.create_study(sampler=sampler_different_seed)
    study_different_seed.optimize(objective, n_trials=15)
    assert any(
        [study.trials[i].params != study_different_seed.trials[i].params for i in range(15)]
    )


# This function is used only in test_reproducible_in_other_process, but declared at top-level
# because local function cannot be pickled, which occurs within multiprocessing.
def run_optimize(
        k: int,
        sampler_name: str,
        sequence_dict: DictProxy,
        hash_dict: DictProxy,
        log_file: str,   # change from Jinglue: added log_file parameter
) -> None:
    try: # change from Jinglue: added try-except block for error handling
        # change from Jinglue: added logging setup
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG,
            format=f'Process {k} - %(asctime)s - %(message)s'
        )

        logging.info(f"Starting process {k}") # change from Jinglue: added logging
        hash_dict[k] = hash("nondeterministic hash")
        logging.info("Hash stored") # change from Jinglue: added logging

        def objective(trial: Trial) -> float:
            a = trial.suggest_float("a", 1, 9)
            b = trial.suggest_float("b", 1, 9, log=True)
            c = trial.suggest_float("c", 1, 9, step=1)
            d = trial.suggest_int("d", 1, 9)
            e = trial.suggest_int("e", 1, 9, log=True)
            f = trial.suggest_int("f", 1, 9, step=2)
            g = trial.suggest_categorical("g", range(1, 10))
            return a + b + c + d + e + f + g

        logging.info("Creating sampler")  # change from Jinglue: added logging
        sampler = sampler_class_with_seed[sampler_name](1)
        logging.info("Creating study")  # change from Jinglue: added logging
        study = optuna.create_study(sampler=sampler)
        logging.info("Starting optimization")  # change from Jinglue: added logging
        study.optimize(objective, n_trials=15)
        sequence_dict[k] = list(study.trials[-1].params.values())
        logging.info("Optimization complete")  # change from Jinglue: added logging

    except Exception as e: # change from Jinglue: added error handling
        import traceback
        logging.error(f"Error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        raise

@pytest.fixture
def unset_seed_in_test(request: SubRequest) -> None:
    # Unset the hashseed at beginning and restore it at end regardless of an exception in the test.
    # See https://docs.pytest.org/en/stable/how-to/fixtures.html#adding-finalizers-directly
    # for details.

    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is not None:
        del os.environ["PYTHONHASHSEED"]

    def restore_seed() -> None:
        if hash_seed is not None:
            os.environ["PYTHONHASHSEED"] = hash_seed

    request.addfinalizer(restore_seed)


@pytest.mark.slow
@parametrize_sampler_name_with_seed
def test_reproducible_in_other_process(sampler_name: str, unset_seed_in_test: None) -> None:
    multiprocessing.set_start_method("spawn", force=True)

    # change from Jinglue: added log file creation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"test_sampler_{timestamp}.log"

    manager = multiprocessing.Manager()
    sequence_dict: DictProxy = manager.dict()
    hash_dict: DictProxy = manager.dict()

    # change from Jinglue: modified process management to track all processes
    processes = []
    for i in range(3):
        p = multiprocessing.Process(
            target=run_optimize,
            args=(i, sampler_name, sequence_dict, hash_dict, log_file)
        )
        processes.append(p)
        p.start()

    # change from Jinglue: modified process checking and added error reporting
    failed = False
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            failed = True

    # change from Jinglue: added log file checking on failure
    if failed:
        print("\nTest failed! Log contents:")
        print("-" * 50)
        try:
            with open(log_file, 'r') as f:
                print(f.read())
        except Exception as e:
            print(f"Error reading log file: {str(e)}")
        print("-" * 50)
        os.remove(log_file)  # Clean up log file
        raise RuntimeError("One or more processes failed. See log contents above.")

    os.remove(log_file)  # change from Jinglue: added log file cleanup

    # Rest of the assertions...
    assert not (hash_dict[0] == hash_dict[1] == hash_dict[2]), "Hashes are expected to be different"
    assert sequence_dict[0] == sequence_dict[1] == sequence_dict[2], "Sequences are expected to be same"


@pytest.mark.parametrize("n_jobs", [1, 2])
@parametrize_relative_sampler
def test_cache_is_invalidated(
    n_jobs: int, relative_sampler_class: Callable[[], BaseSampler]
) -> None:
    sampler = relative_sampler_class()
    original_before_trial = sampler.before_trial

    def mock_before_trial(study: Study, trial: FrozenTrial) -> None:
        assert study._thread_local.cached_all_trials is None
        original_before_trial(study, trial)

    with patch.object(sampler, "before_trial", side_effect=mock_before_trial):
        study = optuna.study.create_study(sampler=sampler)

        def objective(trial: Trial) -> float:
            assert trial._relative_params is None

            trial.suggest_float("x", -10, 10)
            trial.suggest_float("y", -10, 10)
            assert trial._relative_params is not None
            return -1

        study.optimize(objective, n_trials=10, n_jobs=n_jobs)