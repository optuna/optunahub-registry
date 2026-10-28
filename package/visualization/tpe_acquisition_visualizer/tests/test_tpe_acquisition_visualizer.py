from __future__ import annotations

import matplotlib.pyplot as plt
import optuna
from optuna.samplers._tpe import sampler as tpe_sampler_module
from optuna.samplers._tpe.sampler import default_gamma
import pytest

from .. import _resolve_gamma
from .. import TPEAcquisitionVisualizer


def _objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 50)
    return x**2


def test_plot_works_with_independent_tpe() -> None:
    visualizer = TPEAcquisitionVisualizer()
    sampler = optuna.samplers.TPESampler(multivariate=False, n_startup_trials=5, seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=10, callbacks=[visualizer])

    assert set(visualizer.log_objects) == set(range(10))
    fig = visualizer.plot(study, 8, "x")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_infeasible_trials_are_not_in_below_group_with_constraints() -> None:
    # Minimizing -x favors large x, but x > 10 violates the constraint, so the "below"
    # (good) group must contain only feasible trials once constraints are taken into account.
    def constraints(trial: optuna.trial.FrozenTrial) -> list[float]:
        return [trial.params["x"] - 10.0]

    visualizer = TPEAcquisitionVisualizer()
    sampler = optuna.samplers.TPESampler(
        multivariate=False, n_startup_trials=5, seed=0, constraints_func=constraints
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda t: -t.suggest_float("x", -10, 50), n_trials=15, callbacks=[visualizer])

    below = visualizer.log_objects[14]["below_trials"]
    assert below and all(t.params["x"] <= 10.0 for t in below)


def test_callback_uses_multi_objective_gamma() -> None:
    # Plotting is single-objective only, but the callback should split trials with the
    # same gamma as TPESampler. Since Optuna v5, multi-objective studies use a gamma
    # without the cap of 25, so the two rules differ for more than 250 trials.
    visualizer = TPEAcquisitionVisualizer()
    sampler = optuna.samplers.TPESampler(multivariate=False, n_startup_trials=5, seed=0)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.optimize(
        lambda t: (t.suggest_float("x", -5, 5) ** 2, (t.params["x"] - 2) ** 2),
        n_trials=10,
        callbacks=[visualizer],
    )
    assert set(visualizer.log_objects) == set(range(10))

    # Once TPE has sampled, it sets its own gamma; a fresh sampler leaves it unresolved,
    # which is the case the callback has to handle right after the startup trials.
    expected = getattr(tpe_sampler_module, "default_gamma_multiobjective", default_gamma)
    assert _resolve_gamma(optuna.samplers.TPESampler(), study)(300) == expected(300)


def test_multivariate_tpe_raises() -> None:
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=0)
    study = optuna.create_study(sampler=sampler)
    with pytest.raises(ValueError, match="multivariate"):
        study.optimize(_objective, n_trials=1, callbacks=[TPEAcquisitionVisualizer()])


@pytest.mark.skipif(
    not hasattr(optuna.samplers.TPESampler, "_is_multivariate"),
    reason="Before Optuna v5, the default TPESampler is not multivariate.",
)
def test_default_tpe_raises_for_single_objective() -> None:
    # Since Optuna v5, the default TPESampler uses multivariate TPE for single-objective
    # studies, so the callback must not silently plot univariate densities.
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    with pytest.raises(ValueError, match="multivariate"):
        study.optimize(_objective, n_trials=1, callbacks=[TPEAcquisitionVisualizer()])
