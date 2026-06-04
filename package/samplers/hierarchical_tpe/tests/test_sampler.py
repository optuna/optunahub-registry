from __future__ import annotations

import logging

import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.samplers import TPESampler
from optuna.samplers._tpe.sampler import _split_trials
from optuna.trial import TrialState
import optunahub
import pytest


_module = optunahub.load_local_module(
    package="samplers/hierarchical_tpe", registry_root="package/"
)
HierarchicalTPESampler = _module.HierarchicalTPESampler
_TreeBranchClassifier = _module.sampler._TreeBranchClassifier


@pytest.fixture(autouse=True)
def _silence_optuna() -> None:
    """Quiet Optuna's logger and enable propagation so caplog can capture its records."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Optuna's logger does not propagate by default, so caplog cannot see its records.
    optuna.logging.enable_propagation()


def conditional_objective(trial: optuna.Trial) -> float:
    """One-level conditional objective: the optimizer selects which leaf to request.

    Args:
        trial: The Optuna trial to evaluate.

    Returns:
        The objective value to minimize.
    """
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam"])
    if optimizer == "sgd":
        lr = trial.suggest_float("lr", 1e-4, 1.0, log=True)
        return (lr - 0.1) ** 2
    beta = trial.suggest_float("beta", 0.8, 0.999)
    return (beta - 0.9) ** 2


def two_level_objective(trial: optuna.Trial) -> float:
    """Two-level conditional objective: ``x -> (n | m) -> leaf``.

    Args:
        trial: The Optuna trial to evaluate.

    Returns:
        The objective value to minimize.
    """
    x = trial.suggest_categorical("x", [True, False])
    if x:
        n = trial.suggest_categorical("n", [True, False])
        return trial.suggest_float("a" if n else "b", -5, 5) ** 2
    m = trial.suggest_categorical("m", [True, False])
    return trial.suggest_float("c" if m else "d", -5, 5) ** 2 + 1


def non_conditional_objective(trial: optuna.Trial) -> float:
    """Non-conditional objective with two always-present parameters.

    Args:
        trial: The Optuna trial to evaluate.

    Returns:
        The objective value to minimize.
    """
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


def test_runs_with_default_classifier() -> None:
    """The sampler optimizes a conditional objective using the learned classifier."""
    sampler = HierarchicalTPESampler(seed=0, n_startup_trials=8)
    study = optuna.create_study(sampler=sampler)
    study.optimize(conditional_objective, n_trials=50)
    assert len(study.trials) == 50
    assert study.best_value < 0.05


def test_runs_with_two_level_hierarchy() -> None:
    """The sampler runs on a two-level conditional hierarchy."""
    sampler = HierarchicalTPESampler(seed=1, n_startup_trials=8)
    study = optuna.create_study(sampler=sampler)
    study.optimize(two_level_objective, n_trials=60)
    assert len(study.trials) == 60


def test_conditional_fn_path_avoids_independent_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An exact ``conditional_fn`` never triggers the independent-sampling fallback.

    Args:
        caplog: Pytest fixture capturing log records.
    """

    def conditional_fn(params: dict[str, object]) -> list[str]:
        """Exact map for ``conditional_objective``.

        Args:
            params: The parameter values chosen so far.

        Returns:
            The names of the parameters requested next.
        """
        return ["lr"] if params["optimizer"] == "sgd" else ["beta"]

    sampler = HierarchicalTPESampler(seed=0, n_startup_trials=8, conditional_fn=conditional_fn)
    study = optuna.create_study(sampler=sampler)
    with caplog.at_level(logging.WARNING, logger="optuna"):
        study.optimize(conditional_objective, n_trials=50)
    assert "sampled independently" not in caplog.text
    assert study.best_value < 0.05


def test_equivalent_to_multivariate_tpe_without_conditionals() -> None:
    """With no conditional structure the sampler matches TPESampler(group=True) exactly."""
    hier = HierarchicalTPESampler(seed=42, n_startup_trials=10)
    base = TPESampler(seed=42, n_startup_trials=10, multivariate=True, group=True)
    study_hier = optuna.create_study(sampler=hier)
    study_base = optuna.create_study(sampler=base)
    study_hier.optimize(non_conditional_objective, n_trials=40)
    study_base.optimize(non_conditional_objective, n_trials=40)
    assert [t.params for t in study_hier.trials] == [t.params for t in study_base.trials]


def test_seed_reproducibility() -> None:
    """Two samplers with the same seed produce identical parameter sequences."""

    def run() -> list[dict[str, object]]:
        """Run a seeded study and return its parameter sequence.

        Returns:
            The parameters of every trial, in order.
        """
        sampler = HierarchicalTPESampler(seed=7, n_startup_trials=8)
        study = optuna.create_study(sampler=sampler)
        study.optimize(conditional_objective, n_trials=40)
        return [t.params for t in study.trials]

    assert run() == run()


def test_hierarchy_is_inferred() -> None:
    """The inferred hierarchy roots at the optimizer group with the leaves as children."""
    sampler = HierarchicalTPESampler(seed=0, n_startup_trials=4)
    study = optuna.create_study(sampler=sampler)
    study.optimize(conditional_objective, n_trials=30)

    trial = study.ask()
    sampler.infer_relative_search_space(study, trial)
    trials = study.get_trials(deepcopy=False)
    hierarchy = sampler._determine_hierarchy(trials)
    groups = [set(g) for g in sampler._search_space_group.search_spaces]

    roots = [i for i, parent in enumerate(hierarchy) if parent is None]
    assert len(roots) == 1
    assert groups[roots[0]] == {"optimizer"}
    for i, group in enumerate(groups):
        if group in ({"lr"}, {"beta"}):
            assert hierarchy[i] == roots[0]


def test_hierarchy_is_cached() -> None:
    """Re-determining the hierarchy with unchanged trials reuses the cached object."""
    sampler = HierarchicalTPESampler(seed=0, n_startup_trials=4)
    study = optuna.create_study(sampler=sampler)
    study.optimize(conditional_objective, n_trials=20)
    trial = study.ask()
    sampler.infer_relative_search_space(study, trial)
    trials = study.get_trials(deepcopy=False)
    first = sampler._determine_hierarchy(trials)
    second = sampler._determine_hierarchy(trials)
    assert first is second


def test_multi_objective_runs() -> None:
    """The sampler runs on a multi-objective conditional study."""

    def mo_objective(trial: optuna.Trial) -> tuple[float, float]:
        """Two-objective conditional objective.

        Args:
            trial: The Optuna trial to evaluate.

        Returns:
            A pair of objective values to minimize.
        """
        opt = trial.suggest_categorical("opt", ["sgd", "adam"])
        value = trial.suggest_float("lr" if opt == "sgd" else "beta", 0.0, 1.0)
        return value, (value - 0.5) ** 2

    sampler = HierarchicalTPESampler(seed=2, n_startup_trials=8)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.optimize(mo_objective, n_trials=40)
    assert len(study.best_trials) >= 1


def test_constraints_are_respected() -> None:
    """The sampler honors constraints and approaches the constrained optimum."""

    def constrained_objective(trial: optuna.Trial) -> float:
        """Conditional objective with a stored inequality constraint.

        Args:
            trial: The Optuna trial to evaluate.

        Returns:
            The objective value to minimize.
        """
        opt = trial.suggest_categorical("opt", ["sgd", "adam"])
        value = trial.suggest_float("lr" if opt == "sgd" else "beta", 0.0, 1.0)
        trial.set_user_attr("constraint", [value - 0.8])
        return (value - 0.3) ** 2

    def constraints(trial: optuna.trial.FrozenTrial) -> list[float]:
        """Return the stored constraint values for a trial.

        Args:
            trial: The trial to read the constraint from.

        Returns:
            The constraint values (feasible when <= 0).
        """
        return trial.user_attrs["constraint"]

    sampler = HierarchicalTPESampler(seed=3, n_startup_trials=8, constraints_func=constraints)
    study = optuna.create_study(sampler=sampler)
    study.optimize(constrained_objective, n_trials=60)
    feasible = [t for t in study.trials if t.user_attrs["constraint"][0] <= 0]
    assert len(feasible) > 0
    best_feasible = min(t.value for t in feasible)
    assert best_feasible < 0.02


def test_falls_back_to_tpe_when_not_grouped(caplog: pytest.LogCaptureFixture) -> None:
    """Disabling ``multivariate`` forces ``group=False`` and logs the TPE fallback.

    Args:
        caplog: Pytest fixture capturing log records.
    """
    sampler = HierarchicalTPESampler(seed=0, n_startup_trials=8, multivariate=False)
    assert sampler._group is False
    study = optuna.create_study(sampler=sampler)
    with caplog.at_level(logging.INFO, logger="optuna"):
        study.optimize(conditional_objective, n_trials=30)
    assert "falling back to the standard TPESampler" in caplog.text


def test_tree_classifier_refits_on_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    """The cached tree is reused until a contradictory trial forces a logged refit.

    Args:
        caplog: Pytest fixture capturing log records.
    """
    classifier = _TreeBranchClassifier()
    features = np.array([[0.0], [1.0]])
    targets = np.array([0, 1])
    classifier._fit_if_needed(features, targets, index=0, n_trials=2)
    tree_before = classifier._trees[0]

    # Same number of trials -> cache hit, no refit.
    classifier._fit_if_needed(features, targets, index=0, n_trials=2)
    assert classifier._trees[0] is tree_before

    # A new, contradictory observation (feature 0.0 now maps to class 1) -> refit + warning.
    new_features = np.array([[0.0], [1.0], [0.0]])
    new_targets = np.array([0, 1, 1])
    with caplog.at_level(logging.WARNING, logger="optuna"):
        classifier._fit_if_needed(new_features, new_targets, index=0, n_trials=3)
    assert "Refitting the branch classifier" in caplog.text

    classifier.reset()
    assert classifier._trees == {}


def test_conditional_decomposition_recovers_base_tpe() -> None:
    """Scoring a group conditionally recovers base TPE on the joint, unlike the marginal.

    Modeling ``{x, a}`` as ``x`` then ``a | x`` (root ratio + conditional child ratio, where the
    conditional is ``joint - marginal``) must equal base TPE's joint ``{x, a}`` ratio exactly,
    while the marginal child ratio differs when ``x`` and ``a`` are correlated. This is the
    identity the conditional acquisition relies on.
    """

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_categorical("x", ["p", "q"])
        a = trial.suggest_float("a", -3, 3)
        return (a - (1.0 if x == "p" else -1.0)) ** 2

    sampler = TPESampler(seed=0, multivariate=True, n_startup_trials=10)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=120)
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    below, above = _split_trials(study, trials, sampler._gamma(len(trials)), False)

    space_x = {"x": CategoricalDistribution(["p", "q"])}
    space_a = {"a": FloatDistribution(-3, 3)}
    space_xa = {"x": space_x["x"], "a": space_a["a"]}
    joint_b = sampler._build_parzen_estimator(study, space_xa, below, handle_below=True)
    joint_a = sampler._build_parzen_estimator(study, space_xa, above, handle_below=False)
    x_b = sampler._build_parzen_estimator(study, space_x, below, handle_below=True)
    x_a = sampler._build_parzen_estimator(study, space_x, above, handle_below=False)
    a_b = sampler._build_parzen_estimator(study, space_a, below, handle_below=True)
    a_a = sampler._build_parzen_estimator(study, space_a, above, handle_below=False)

    test = joint_b.sample(np.random.RandomState(1), 200)
    base = joint_b.log_pdf(test) - joint_a.log_pdf(test)
    root_x = x_b.log_pdf({"x": test["x"]}) - x_a.log_pdf({"x": test["x"]})
    conditional_a = base - root_x  # log l(a | x) - log g(a | x), via joint - marginal
    marginal_a = a_b.log_pdf({"a": test["a"]}) - a_a.log_pdf({"a": test["a"]})

    # x -> a|x recovers base TPE on the joint exactly.
    np.testing.assert_allclose(root_x + conditional_a, base, atol=1e-10)
    # The conditional child differs from the marginal child (correlation is captured).
    assert np.mean(np.abs(conditional_a - marginal_a)) > 1e-3
