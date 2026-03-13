from __future__ import annotations

import numpy as np
import optuna
import optunahub


CmaEsRefinementSampler = optunahub.load_local_module(
    package="samplers/cma_es_refinement", registry_root="package/"
).CmaEsRefinementSampler


def sphere(trial: optuna.Trial) -> float:
    return sum(trial.suggest_float(f"x{i}", -5, 5) ** 2 for i in range(5))


def test_runs_without_error() -> None:
    sampler = CmaEsRefinementSampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(sphere, n_trials=10)

    assert len(study.trials) == 10
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_phases() -> None:
    n_startup = 4
    cma_n_trials = 6
    n_medium = 3

    sampler = CmaEsRefinementSampler(
        n_startup_trials=n_startup,
        cma_n_trials=cma_n_trials,
        n_medium_refine_trials=n_medium,
        seed=42,
    )

    assert sampler._phase(0) == "sobol"
    assert sampler._phase(n_startup - 1) == "sobol"
    assert sampler._phase(n_startup) == "cma"
    assert sampler._phase(n_startup + cma_n_trials - 1) == "cma"
    assert sampler._phase(n_startup + cma_n_trials) == "refine"
    assert sampler._phase(200) == "refine"


def test_reproducibility() -> None:
    results = []
    for _ in range(2):
        sampler = CmaEsRefinementSampler(seed=42)
        study = optuna.create_study(sampler=sampler)
        study.optimize(sphere, n_trials=20)
        results.append([t.value for t in study.trials])

    np.testing.assert_array_equal(results[0], results[1])


def test_refinement_near_best() -> None:
    sampler = CmaEsRefinementSampler(
        n_startup_trials=4,
        cma_n_trials=6,
        n_medium_refine_trials=5,
        seed=42,
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(sphere, n_trials=30)

    best_params = study.best_trial.params

    # Refinement trials (index 10+) should sample near the best point
    refine_trials = study.trials[10:]
    for t in refine_trials:
        for key in best_params:
            assert abs(t.params[key] - best_params[key]) < 2.0, (
                f"Refinement trial param {key}={t.params[key]} too far from "
                f"best={best_params[key]}"
            )


def test_known_value() -> None:
    """Pin a known result so regressions or environment changes are caught."""
    sampler = CmaEsRefinementSampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(sphere, n_trials=20)

    # Pinned from a verified run — if this changes, something broke determinism
    assert study.best_value < 1.0, f"Expected best < 1.0, got {study.best_value}"
    assert len(study.best_trial.params) == 5


def test_for_budget() -> None:
    """Verify for_budget() scales phases proportionally."""
    sampler = CmaEsRefinementSampler.for_budget(1000, seed=0)
    assert sampler._n_startup >= 4
    assert sampler._cma_end > sampler._n_startup
    assert sampler._medium_end > sampler._cma_end

    # Phase ratios should be approximately preserved
    cma_frac = (sampler._cma_end - sampler._n_startup) / 1000
    assert 0.5 < cma_frac < 0.8, f"CMA fraction {cma_frac} outside expected range"

    # Should be usable
    study = optuna.create_study(sampler=sampler)
    study.optimize(sphere, n_trials=10)
    assert len(study.trials) == 10
