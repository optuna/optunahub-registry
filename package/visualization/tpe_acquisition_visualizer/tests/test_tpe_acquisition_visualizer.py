from __future__ import annotations

import matplotlib.pyplot as plt
import optuna
import pytest

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
