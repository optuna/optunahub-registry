from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pytest

from .. import BOCSSampler
from .. import sampler as sampler_module
from .._acquisition import maximize_sdp
from .._annealing import maximize_sa


optuna.logging.set_verbosity(optuna.logging.WARNING)

D = 6


def _fast_sampler(seed: int = 0, acquisition: str = "sdp") -> BOCSSampler:
    return BOCSSampler(
        acquisition=acquisition,  # type: ignore[arg-type]
        n_startup_trials=5,
        n_burnin=100,
        n_gibbs_steps=20,
        seed=seed,
    )


def _count_ones(trial: optuna.Trial) -> float:
    return float(sum(trial.suggest_int(f"x{i}", 0, 1) for i in range(D)))


@pytest.mark.parametrize("direction, best", [("maximize", D), ("minimize", 0)])
def test_finds_optimum_of_linear_objective(direction: str, best: int) -> None:
    study = optuna.create_study(direction=direction, sampler=_fast_sampler())
    study.optimize(_count_ones, n_trials=15)
    assert study.best_value == best
    assert all(v in (0, 1) for t in study.trials for v in t.params.values())


def test_supports_binary_categorical_parameters() -> None:
    def objective(trial: optuna.Trial) -> float:
        flags = [trial.suggest_categorical(f"f{i}", ["off", "on"]) for i in range(D)]
        return float(flags.count("on"))

    study = optuna.create_study(direction="maximize", sampler=_fast_sampler())
    study.optimize(objective, n_trials=15)
    assert study.best_value == D
    assert all(v in ("off", "on") for t in study.trials for v in t.params.values())


def test_non_binary_parameters_fall_back_to_independent_sampler() -> None:
    def objective(trial: optuna.Trial) -> float:
        scale = trial.suggest_float("scale", 0.5, 2.0)
        return scale * _count_ones(trial)

    sampler = BOCSSampler(
        n_startup_trials=5, n_burnin=100, n_gibbs_steps=20, seed=0, warn_independent_sampling=False
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)
    assert all(0.5 <= t.params["scale"] <= 2.0 for t in study.trials)
    assert sampler.infer_relative_search_space(study, study.trials[-1]).keys() == {
        f"x{i}" for i in range(D)
    }


def test_is_reproducible_with_seed() -> None:
    def run() -> list[dict[str, int]]:
        study = optuna.create_study(sampler=_fast_sampler(seed=42))
        study.optimize(_count_ones, n_trials=10)
        return [t.params for t in study.trials]

    assert run() == run()


@pytest.mark.parametrize("acquisition", ["sdp", "sa"])
def test_proposes_points_that_maximize_a_quadratic_objective(acquisition: str) -> None:
    rng = np.random.default_rng(0)
    Q = rng.normal(size=(D, D))

    def objective(trial: optuna.Trial) -> float:
        x = np.array([trial.suggest_int(f"x{i}", 0, 1) for i in range(D)])
        return float(x @ Q @ x)

    study = optuna.create_study(
        direction="maximize", sampler=_fast_sampler(acquisition=acquisition)
    )
    study.optimize(objective, n_trials=25)
    all_x = (np.arange(2**D)[:, None] >> np.arange(D)) & 1
    assert study.best_value == pytest.approx(np.einsum("ni,ij,nj->n", all_x, Q, all_x).max())


@pytest.mark.parametrize("acquisition, used, unused", [("sdp", "sdp", "sa"), ("sa", "sa", "sdp")])
def test_acquisition_option_selects_the_optimizer(
    monkeypatch: pytest.MonkeyPatch, acquisition: str, used: str, unused: str
) -> None:
    calls = {"sdp": 0, "sa": 0}

    def counting(name: str, original: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(sampler_module, "maximize_sdp", counting("sdp", maximize_sdp))
    monkeypatch.setattr(sampler_module, "maximize_sa", counting("sa", maximize_sa))
    study = optuna.create_study(sampler=_fast_sampler(acquisition=acquisition))
    study.optimize(_count_ones, n_trials=7)
    assert calls[used] == 2 and calls[unused] == 0


def test_rejects_unknown_acquisition() -> None:
    with pytest.raises(ValueError, match="acquisition"):
        BOCSSampler(acquisition="ei")  # type: ignore[arg-type]


def test_rejects_multi_objective_studies() -> None:
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=_fast_sampler())
    for _ in range(5):
        study.enqueue_trial({f"x{i}": 0 for i in range(D)})
    study.optimize(lambda t: (_count_ones(t), -_count_ones(t)), n_trials=5)
    with pytest.raises(ValueError, match="multi-objective"):
        study.optimize(lambda t: (_count_ones(t), -_count_ones(t)), n_trials=1)
