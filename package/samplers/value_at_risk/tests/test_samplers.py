from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import multiprocessing
from multiprocessing.managers import DictProxy
import os
from typing import Any
from unittest.mock import patch
import warnings

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
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState
import optunahub
import pytest


RobustGPSampler = optunahub.load_local_module(
    package="samplers/value_at_risk", registry_root="package/"
).RobustGPSampler


def get_gp_sampler(
    *, n_startup_trials: int = 0, deterministic_objective: bool = False, seed: int | None = None
) -> RobustGPSampler:  # type: ignore[valid-type]
    sampler = RobustGPSampler(
        n_startup_trials=n_startup_trials,
        seed=seed,
        deterministic_objective=deterministic_objective,
        # NOTE: We set empty dict to avoid the assertion error in the constructor.
        uniform_input_noise_rads={},
    )
    # NOTE: Replace {} with None to unset the noise setting. We set noise in set_noise_to_sampler.
    sampler._uniform_input_noise_rads = None
    # NOTE: Replace these attributes to reduce the execution time of tests.
    sampler._n_preliminary_samples = 512
    sampler._n_local_search = 3
    return sampler


def set_noise_to_sampler(
    sampler: RobustGPSampler,  # type: ignore[valid-type]
    uniform_input_noise_rads: dict[str, float] | None = None,
    normal_input_noise_stdevs: dict[str, float] | None = None,
) -> None:
    assert uniform_input_noise_rads is None or normal_input_noise_stdevs is None
    if uniform_input_noise_rads is not None:
        sampler._uniform_input_noise_rads = uniform_input_noise_rads  # type: ignore[attr-defined]
    elif normal_input_noise_stdevs is not None:
        sampler._normal_input_noise_stdevs = normal_input_noise_stdevs  # type: ignore[attr-defined]
    else:
        assert False, "Should not reach"


parametrize_noise_type = pytest.mark.parametrize("noise_type", ["uniform", "normal"])
parametrize_sampler = pytest.mark.parametrize(
    "sampler_class",
    [
        lambda: get_gp_sampler(n_startup_trials=1),
        lambda: get_gp_sampler(n_startup_trials=1, deterministic_objective=True),
    ],
)
parametrize_relative_sampler = pytest.mark.parametrize(
    "relative_sampler_class",
    [
        lambda: get_gp_sampler(n_startup_trials=1),
        lambda: get_gp_sampler(n_startup_trials=1, deterministic_objective=True),
    ],
)


sampler_class_with_seed: dict[str, Callable[[int], BaseSampler]] = {
    "GPSampler": lambda seed: get_gp_sampler(seed=seed, n_startup_trials=1),
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


@pytest.mark.parametrize(
    "sampler_class,expected_has_rng,expected_has_another_sampler",
    [(lambda: get_gp_sampler(n_startup_trials=1), True, True)],
)
def test_sampler_reseed_rng(
    sampler_class: Callable[[], BaseSampler],
    expected_has_rng: bool,
    expected_has_another_sampler: bool,
) -> None:
    def _extract_attr_name_from_sampler_by_cls(sampler: BaseSampler, cls: Any) -> str | None:
        for name, attr in sampler.__dict__.items():
            if isinstance(attr, cls):
                return name
        return None

    sampler = sampler_class()

    rng_name = _extract_attr_name_from_sampler_by_cls(sampler, LazyRandomState)
    has_rng = rng_name is not None
    assert expected_has_rng == has_rng
    if has_rng:
        rng_name = str(rng_name)
        original_random_state = sampler.__dict__[rng_name].rng.get_state()
        sampler.reseed_rng()
        random_state = sampler.__dict__[rng_name].rng.get_state()
        if not isinstance(sampler, optuna.samplers.CmaEsSampler):
            assert str(original_random_state) != str(random_state)
        else:
            # CmaEsSampler has a RandomState that is not reseed by its reseed_rng method.
            assert str(original_random_state) == str(random_state)

    had_sampler_name = _extract_attr_name_from_sampler_by_cls(sampler, BaseSampler)
    has_another_sampler = had_sampler_name is not None
    assert expected_has_another_sampler == has_another_sampler

    if has_another_sampler:
        had_sampler_name = str(had_sampler_name)
        had_sampler = sampler.__dict__[had_sampler_name]
        had_sampler_rng_name = _extract_attr_name_from_sampler_by_cls(had_sampler, LazyRandomState)
        original_had_sampler_random_state = had_sampler.__dict__[
            had_sampler_rng_name
        ].rng.get_state()
        with patch.object(
            had_sampler,
            "reseed_rng",
            wraps=had_sampler.reseed_rng,
        ) as mock_object:
            sampler.reseed_rng()
            assert mock_object.call_count == 1

        had_sampler = sampler.__dict__[had_sampler_name]
        had_sampler_random_state = had_sampler.__dict__[had_sampler_rng_name].rng.get_state()
        assert str(original_had_sampler_random_state) != str(had_sampler_random_state)


def parametrize_suggest_method(name: str) -> MarkDecorator:
    return pytest.mark.parametrize(
        f"suggest_method_{name},can_be_noisy_{name}",
        [
            (lambda t: t.suggest_float(name, 0, 10), True),
            (lambda t: t.suggest_int(name, 0, 10), False),
            (lambda t: t.suggest_categorical(name, [0, 1, 2]), False),
            (lambda t: t.suggest_float(name, 0, 10, step=0.5), False),
            (lambda t: t.suggest_float(name, 1e-7, 10, log=True), False),
            (lambda t: t.suggest_int(name, 1, 10, log=True), False),
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
    sampler_class: Callable[[], BaseSampler],
    distribution: FloatDistribution,
) -> None:
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
@parametrize_noise_type
def test_sample_relative_numerical(
    relative_sampler_class: Callable[[], BaseSampler],
    x_distribution: FloatDistribution | IntDistribution,
    y_distribution: FloatDistribution | IntDistribution,
    noise_type: str,
) -> None:
    can_x_be_noisy = x_distribution.step is None and not x_distribution.log
    can_y_be_noisy = y_distribution.step is None and not y_distribution.log
    if not can_x_be_noisy and not can_y_be_noisy:
        pytest.skip("Skip because both distributions cannot be noisy.")

    search_space: dict[str, BaseDistribution] = dict(x=x_distribution, y=y_distribution)
    sampler = relative_sampler_class()
    noise_params = {}
    if noise_type == "uniform":
        if can_x_be_noisy:
            noise_params["x"] = 0.1
        elif can_y_be_noisy:
            noise_params["y"] = 0.1
        set_noise_to_sampler(sampler, uniform_input_noise_rads=noise_params)
    elif noise_type == "normal":
        if can_x_be_noisy:
            noise_params["x"] = 0.05
        elif can_y_be_noisy:
            noise_params["y"] = 0.05
        set_noise_to_sampler(sampler, normal_input_noise_stdevs=noise_params)

    study = optuna.study.create_study(sampler=sampler)
    trial = study.ask(search_space)
    study.tell(trial, sum(trial.params.values()))

    def sample() -> list[int | float]:
        params = study.sampler.sample_relative(study, _create_new_trial(study), search_space)
        return [params[name] for name in search_space]

    points = np.array([sample() for _ in range(11)])
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
@parametrize_noise_type
def test_sample_relative_mixed(
    relative_sampler_class: Callable[[], BaseSampler], noise_type: str
) -> None:
    x_distribution = FloatDistribution(-1.0, 1.0)
    search_space: dict[str, BaseDistribution] = dict(
        x=x_distribution, y=CategoricalDistribution([-1, -10, -100])
    )
    sampler = relative_sampler_class()
    set_noise_to_sampler(
        sampler,
        uniform_input_noise_rads=None if noise_type == "normal" else {"x": 0.1},
        normal_input_noise_stdevs=None if noise_type == "uniform" else {"x": 0.05},
    )
    study = optuna.study.create_study(sampler=sampler)
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


class FixedSampler(BaseSampler):
    def __init__(
        self,
        relative_search_space: dict[str, BaseDistribution],
        relative_params: dict[str, Any],
        unknown_param_value: Any,
    ) -> None:
        self.relative_search_space = relative_search_space
        self.relative_params = relative_params
        self.unknown_param_value = unknown_param_value

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return self.relative_search_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self.relative_params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self.unknown_param_value


@parametrize_sampler
@parametrize_noise_type
def test_nan_objective_value(sampler_class: Callable[[], BaseSampler], noise_type: str) -> None:
    sampler = sampler_class()
    set_noise_to_sampler(
        sampler,
        uniform_input_noise_rads={"x": 0.005} if noise_type == "uniform" else None,
        normal_input_noise_stdevs={"x": 0.0025} if noise_type == "normal" else None,
    )
    study = optuna.create_study(sampler=sampler)

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


@parametrize_sampler
@parametrize_noise_type
def test_partial_fixed_sampling(sampler_class: Callable[[], BaseSampler], noise_type: str) -> None:
    sampler = sampler_class()
    set_noise_to_sampler(
        sampler,
        uniform_input_noise_rads={"x": 0.05} if noise_type == "uniform" else None,
        normal_input_noise_stdevs={"x": 0.025} if noise_type == "normal" else None,
    )
    study = optuna.create_study(sampler=sampler)

    def objective(trial: Trial) -> float:
        x = trial.suggest_float("x", -1, 1)
        y = trial.suggest_int("y", -1, 1)
        z = trial.suggest_float("z", -1, 1)
        return x + y + z

    # First trial.
    study.optimize(objective, n_trials=1)

    # Second trial. Here, the parameter ``y`` is fixed as 0.
    fixed_params = {"y": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study.sampler = optuna.samplers.PartialFixedSampler(fixed_params, study.sampler)
    study.optimize(objective, n_trials=1)
    trial_params = study.trials[-1].params
    assert trial_params["y"] == fixed_params["y"]


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
@parametrize_noise_type
def test_single_parameter_objective(
    sampler_class: Callable[[], BaseSampler],
    suggest_method_x: Callable[[Trial], float],
    can_be_noisy_x: bool,
    noise_type: str,
) -> None:
    if not can_be_noisy_x:
        pytest.skip("Skip because search space does not contain noisy params.")

    def objective(trial: Trial) -> float:
        return suggest_method_x(trial)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()
        set_noise_to_sampler(
            sampler,
            uniform_input_noise_rads={"x": 0.5} if noise_type == "uniform" else None,
            normal_input_noise_stdevs={"x": 0.25} if noise_type == "normal" else None,
        )

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=11)

    assert len(study.trials) == 11
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


@parametrize_sampler
@parametrize_suggest_method("x")
@parametrize_suggest_method("y")
@parametrize_noise_type
def test_combination_of_different_distributions_objective(
    sampler_class: Callable[[], BaseSampler],
    suggest_method_x: Callable[[Trial], float],
    can_be_noisy_x: bool,
    suggest_method_y: Callable[[Trial], float],
    can_be_noisy_y: bool,
    noise_type: str,
) -> None:
    if not can_be_noisy_x and not can_be_noisy_y:
        pytest.skip("Skip because search space does not contain noisy params.")

    def objective(trial: Trial) -> float:
        return suggest_method_x(trial) + suggest_method_y(trial)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()
        noise_parmas = {}
        if noise_type == "uniform":
            if can_be_noisy_x:
                noise_parmas["x"] = 0.5
            if can_be_noisy_y:
                noise_parmas["y"] = 0.5
            set_noise_to_sampler(sampler, uniform_input_noise_rads=noise_parmas)
        elif noise_type == "normal":
            if can_be_noisy_x:
                noise_parmas["x"] = 0.25
            if can_be_noisy_y:
                noise_parmas["y"] = 0.25
            set_noise_to_sampler(sampler, normal_input_noise_stdevs=noise_parmas)

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=11)

    assert len(study.trials) == 11
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
@parametrize_noise_type
def test_dynamic_range_objective(
    sampler_class: Callable[[], BaseSampler], second_low: int, second_high: int, noise_type: str
) -> None:
    def objective(trial: Trial, low: int, high: int) -> float:
        v = trial.suggest_float("x", low, high)
        v += trial.suggest_int("y", low, high)
        return v

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()
        set_noise_to_sampler(
            sampler,
            uniform_input_noise_rads={"x": 0.25} if noise_type == "uniform" else None,
            normal_input_noise_stdevs={"x": 0.125} if noise_type == "normal" else None,
        )

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(lambda t: objective(t, 0, 10), n_trials=10)
    study.optimize(lambda t: objective(t, second_low, second_high), n_trials=10)

    assert len(study.trials) == 20
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


# We add tests for constant objective functions to ensure the reproducibility of sorting.
@parametrize_sampler_with_seed
@pytest.mark.slow
@pytest.mark.parametrize("objective_func", [lambda *args: sum(args), lambda *args: 0.0])
@parametrize_noise_type
def test_reproducible(
    sampler_class: Callable[[int], BaseSampler], objective_func: Any, noise_type: str
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

    def _get_sampler(seed: int) -> BaseSampler:
        sampler = sampler_class(seed)
        set_noise_to_sampler(
            sampler,
            uniform_input_noise_rads={"a": 0.25} if noise_type == "uniform" else None,
            normal_input_noise_stdevs={"a": 0.125} if noise_type == "normal" else None,
        )
        return sampler

    n_trials = 12
    study = optuna.create_study(sampler=_get_sampler(1))
    study.optimize(objective, n_trials=n_trials)

    study_same_seed = optuna.create_study(sampler=_get_sampler(1))
    study_same_seed.optimize(objective, n_trials=n_trials)
    for i in range(n_trials):
        assert study.trials[i].params == study_same_seed.trials[i].params

    study_different_seed = optuna.create_study(sampler=_get_sampler(2))
    study_different_seed.optimize(objective, n_trials=n_trials)
    assert any(
        [study.trials[i].params != study_different_seed.trials[i].params for i in range(n_trials)]
    )


@pytest.mark.slow
@parametrize_sampler_with_seed
@parametrize_noise_type
def test_reseed_rng_change_sampling(
    sampler_class: Callable[[int], BaseSampler], noise_type: str
) -> None:
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
    set_noise_to_sampler(
        sampler,
        uniform_input_noise_rads={"a": 0.25} if noise_type == "uniform" else None,
        normal_input_noise_stdevs={"a": 0.125} if noise_type == "normal" else None,
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=15)

    sampler_different_seed = sampler_class(1)
    set_noise_to_sampler(
        sampler_different_seed,
        uniform_input_noise_rads={"a": 0.25} if noise_type == "uniform" else None,
        normal_input_noise_stdevs={"a": 0.125} if noise_type == "normal" else None,
    )
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
    noise_type: str,
) -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_float("a", 1, 9)
        b = trial.suggest_float("b", 1, 9, log=True)
        c = trial.suggest_float("c", 1, 9, step=1)
        d = trial.suggest_int("d", 1, 9)
        e = trial.suggest_int("e", 1, 9, log=True)
        f = trial.suggest_int("f", 1, 9, step=2)
        g = trial.suggest_categorical("g", range(1, 10))
        return a + b + c + d + e + f + g

    hash_dict[k] = hash("nondeterministic hash")
    sampler = sampler_class_with_seed[sampler_name](1)
    set_noise_to_sampler(
        sampler,
        uniform_input_noise_rads={"a": 0.25} if noise_type == "uniform" else None,
        normal_input_noise_stdevs={"a": 0.125} if noise_type == "normal" else None,
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=12)
    sequence_dict[k] = list(study.trials[-1].params.values())


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
@parametrize_noise_type
def test_reproducible_in_other_process(
    sampler_name: str, unset_seed_in_test: None, noise_type: str
) -> None:
    # This test should be tested without `PYTHONHASHSEED`. However, some tool such as tox
    # set the environmental variable "PYTHONHASHSEED" by default.
    # To do so, this test calls a finalizer: `unset_seed_in_test`.

    # Multiprocessing supports three way to start a process.
    # We use `spawn` option to create a child process as a fresh python process.
    # For more detail, see https://github.com/optuna/optuna/pull/3187#issuecomment-997673037.
    multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    sequence_dict: DictProxy = manager.dict()
    hash_dict: DictProxy = manager.dict()
    for i in range(3):
        p = multiprocessing.Process(
            target=run_optimize, args=(i, sampler_name, sequence_dict, hash_dict, noise_type)
        )
        p.start()
        p.join()

    # Hashes are expected to be different because string hashing is nondeterministic per process.
    assert not (hash_dict[0] == hash_dict[1] == hash_dict[2])
    # But the sequences are expected to be the same.
    assert sequence_dict[0] == sequence_dict[1] == sequence_dict[2]


@pytest.mark.parametrize("n_jobs", [1, 2])
@parametrize_relative_sampler
@parametrize_noise_type
def test_trial_relative_params(
    n_jobs: int, relative_sampler_class: Callable[[], BaseSampler], noise_type: str
) -> None:
    # TODO(nabenabe): Consider moving this test to study.
    sampler = relative_sampler_class()
    set_noise_to_sampler(
        sampler,
        uniform_input_noise_rads={"x": 0.1, "y": 0.2} if noise_type == "uniform" else None,
        normal_input_noise_stdevs={"x": 0.05, "y": 0.1} if noise_type == "normal" else None,
    )
    study = optuna.study.create_study(sampler=sampler)

    def objective(trial: Trial) -> float:
        assert trial._relative_params is None

        trial.suggest_float("x", -10, 10)
        trial.suggest_float("y", -10, 10)
        assert trial._relative_params is not None
        return -1

    study.optimize(objective, n_trials=11, n_jobs=n_jobs)
