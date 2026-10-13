"""Unit tests for the SVM hyperparameter optimization benchmark."""

from __future__ import annotations

import optuna
import pytest

from .._svm_hpo import Problem


optuna.logging.set_verbosity(optuna.logging.WARNING)

# A cheap configuration that is known to work well on all datasets.
GOOD_PARAMS = {"C": 1.0, "gamma": 0.01, "kernel": "rbf"}


@pytest.mark.parametrize("dataset_name", Problem.available_dataset_names)
def test_evaluate_returns_error_in_unit_interval(dataset_name: str) -> None:
    problem = Problem(dataset_name=dataset_name, n_splits=3, seed=0)
    error = problem.evaluate(GOOD_PARAMS)
    assert isinstance(error, float)
    assert 0.0 <= error <= 1.0


def test_good_configuration_beats_chance() -> None:
    # Breast cancer is binary, so a sensible SVM must clearly beat 50% error.
    problem = Problem(dataset_name="breast_cancer", seed=0)
    assert problem.evaluate(GOOD_PARAMS) < 0.1


def test_search_space_and_directions() -> None:
    problem = Problem(seed=0)
    space = problem.search_space
    assert set(space) == {"C", "gamma", "kernel", "degree"}
    assert space["C"].log and space["gamma"].log  # type: ignore[attr-defined]
    assert set(space["kernel"].choices) == {  # type: ignore[attr-defined]
        "linear",
        "rbf",
        "poly",
        "sigmoid",
    }
    assert problem.directions == [optuna.study.StudyDirection.MINIMIZE]


def test_search_space_returns_a_fresh_copy() -> None:
    problem = Problem(seed=0)
    problem.search_space.pop("C")
    assert "C" in problem.search_space


@pytest.mark.parametrize("kernel", ["linear", "rbf", "sigmoid"])
def test_degree_is_not_sampled_for_non_poly_kernels(kernel: str) -> None:
    problem = Problem(n_splits=2, seed=0)
    study = optuna.create_study(directions=problem.directions)
    study.enqueue_trial({"C": 1.0, "gamma": 0.01, "kernel": kernel})
    study.optimize(problem, n_trials=1)
    assert "degree" not in study.trials[0].params


def test_degree_is_sampled_for_poly_kernel() -> None:
    problem = Problem(n_splits=2, seed=0)
    study = optuna.create_study(directions=problem.directions)
    study.enqueue_trial({"C": 1.0, "gamma": 0.01, "kernel": "poly", "degree": 2})
    study.optimize(problem, n_trials=1)
    assert study.trials[0].params["degree"] == 2


def test_evaluate_without_degree_uses_default_for_poly() -> None:
    problem = Problem(n_splits=2, seed=0)
    error = problem.evaluate({"C": 1.0, "gamma": 0.01, "kernel": "poly"})
    assert 0.0 <= error <= 1.0


def test_optimize_runs_end_to_end() -> None:
    problem = Problem(dataset_name="wine", n_splits=3, seed=0)
    study = optuna.create_study(
        directions=problem.directions, sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(problem, n_trials=10)
    assert len(study.trials) == 10
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert 0.0 <= study.best_value <= 1.0


def test_same_seed_gives_same_result() -> None:
    params = {"C": 10.0, "gamma": 0.1, "kernel": "rbf"}
    first = Problem(seed=123).evaluate(params)
    second = Problem(seed=123).evaluate(params)
    assert first == second


def test_reseed_changes_folds_reproducibly() -> None:
    params = {"C": 10.0, "gamma": 0.1, "kernel": "rbf"}
    problem = Problem(seed=1)
    errors_seed_1 = problem.evaluate(params)
    problem.reseed(2)
    errors_seed_2 = problem.evaluate(params)
    problem.reseed(1)
    assert problem.evaluate(params) == errors_seed_1
    # Different fold splits almost always give a slightly different score.
    assert errors_seed_1 != errors_seed_2


def test_invalid_dataset_name_raises() -> None:
    with pytest.raises(ValueError, match="dataset_name"):
        Problem(dataset_name="iris")


def test_invalid_n_splits_raises() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        Problem(n_splits=1)


def test_invalid_kernel_raises() -> None:
    problem = Problem(seed=0)
    with pytest.raises(ValueError, match="kernel"):
        problem.evaluate({"C": 1.0, "gamma": 0.1, "kernel": "precomputed"})
