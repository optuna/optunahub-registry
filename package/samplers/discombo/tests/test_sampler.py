from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial
import optunahub
import pytest


module = optunahub.load_local_module(package="samplers/discombo", registry_root="package/")
DisCoMBOSampler = module.DisCoMBOSampler
FAST = {"n_startup_trials": 8, "batch_size": 4, "stratified_pool_size": 200}

optuna.logging.set_verbosity(optuna.logging.WARNING)


class RecordingSampler(DisCoMBOSampler):  # type: ignore[misc, valid-type]
    """Records the relative proposals to check that the trials use them."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.proposed: dict[int, dict[str, Any]] = {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        params = super().sample_relative(study, trial, search_space)
        if params:
            self.proposed[trial.number] = params
        return params


def mixed_objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.0, 1.0, step=0.05)
    layers = trial.suggest_int("layers", 1, 4)
    units = trial.suggest_int("units", 16, 512, log=True)
    batch = trial.suggest_int("batch", 8, 128, step=8)
    optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam", None])
    return (
        (np.log10(lr) + 2) ** 2
        + (momentum - 0.9) ** 2
        + 0.1 * abs(layers - 3)
        + 0.001 * abs(units - 100)
        + 0.001 * abs(batch - 64)
        - 0.2 * (optimizer == "adam")
    )


def test_trials_use_the_proposals_of_discombo() -> None:
    sampler = RecordingSampler(seed=0, **FAST)
    study = optuna.create_study(sampler=sampler)
    study.optimize(mixed_objective, n_trials=24)
    assert set(sampler.proposed) == set(range(8, 24))
    for number, params in sampler.proposed.items():
        assert study.trials[number].params == params


@pytest.mark.parametrize(
    "dist",
    [
        FloatDistribution(-1.0, 1.0),
        FloatDistribution(1e-5, 1.0, log=True),
        FloatDistribution(-1.0, 1.0, step=0.1),
        IntDistribution(0, 5),
        IntDistribution(-10, 100, step=5),
        IntDistribution(1, 4096, log=True),
        CategoricalDistribution(["a", 1, None, 2.5, True]),
    ],
)
def test_values_are_mapped_into_the_distribution(dist: BaseDistribution) -> None:
    definition = module.sampler._to_definition(dist)
    rng = np.random.RandomState(0)
    for _ in range(200):
        if "allowed" in definition:
            value = definition["allowed"][rng.randint(len(definition["allowed"]))]
        else:  # DisCoMBO clamps continuous values to the bounds (in log space for log)
            lo, hi = definition["min"], definition["max"]
            if definition["is_log"]:
                value = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                value = float(rng.uniform(lo, hi))
        optuna_value = module.sampler._to_optuna(value, dist)
        assert dist._contains(dist.to_internal_repr(optuna_value))


def test_seed_makes_studies_reproducible() -> None:
    def run(seed: int) -> list[dict[str, Any]]:
        study = optuna.create_study(sampler=DisCoMBOSampler(seed=seed, **FAST))
        study.optimize(mixed_objective, n_trials=16)
        return [t.params for t in study.trials]

    assert run(0) == run(0)
    assert run(0) != run(1)


def test_minimization_and_maximization_are_equivalent() -> None:
    def run(direction: str) -> list[dict[str, Any]]:
        sign = 1.0 if direction == "minimize" else -1.0
        study = optuna.create_study(direction=direction, sampler=DisCoMBOSampler(seed=3, **FAST))
        study.optimize(lambda t: sign * mixed_objective(t), n_trials=16)
        return [t.params for t in study.trials]

    assert run("minimize") == run("maximize")


def test_failed_pruned_conditional_and_enqueued_trials() -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -5.0, 5.0)
        kind = trial.suggest_categorical("kind", ["a", "b"])
        if kind == "a":
            x += trial.suggest_float("only_a", 0.0, 1.0)  # outside the intersection search space
        if trial.number % 7 == 3:
            raise RuntimeError("failed evaluation")
        if trial.number % 7 == 5:
            raise optuna.TrialPruned()
        return x**2

    study = optuna.create_study(sampler=DisCoMBOSampler(seed=0, **FAST))
    study.enqueue_trial({"x": 0.0, "kind": "b"})
    study.optimize(objective, n_trials=24, catch=(RuntimeError,))
    states = [t.state for t in study.trials]
    assert states.count(optuna.trial.TrialState.FAIL) > 0
    assert states.count(optuna.trial.TrialState.PRUNED) > 0
    assert study.best_value == 0.0


def test_fixed_search_space_and_parallel_trials() -> None:
    search_space = {"x": FloatDistribution(-5.0, 5.0), "n": IntDistribution(0, 20)}
    sampler = DisCoMBOSampler(search_space=search_space, seed=0, **FAST)
    study = optuna.create_study(sampler=sampler)
    study.optimize(
        lambda t: t.suggest_float("x", -5.0, 5.0) ** 2 + t.suggest_int("n", 0, 20),
        n_trials=20,
        n_jobs=2,
    )
    assert len(study.trials) == 20


def test_interventions_report_their_disco_scores() -> None:
    sampler = DisCoMBOSampler(seed=0, **FAST)
    sampler.intervene({"lr": 1e-2, "optimizer": "adam", "layers": 3}, trial_number=12)
    sampler.intervene({"momentum": {"dist": "gauss", "parameters": [0.9, 0.05]}}, trial_number=16)
    sampler.intervene({"unknown": 1.0}, trial_number=0)
    sampler.intervene({"optimizer": "not a choice"}, trial_number=0)
    study = optuna.create_study(sampler=sampler)
    with pytest.warns(UserWarning) as warned:
        study.optimize(mixed_objective, n_trials=24)
    messages = [str(w.message) for w in warned]
    assert any("'unknown'" in m for m in messages)
    assert any("'not a choice'" in m for m in messages)
    records = sampler.get_disco_scores()
    assert [r["trial_number"] for r in records] == [12, 16]
    assert records[0]["beliefs"] == {"lr": np.log(1e-2), "optimizer": 1, "layers": 2}
    assert all(0.0 <= r["disco_score"] <= 1.0 for r in records)
    assert all(len(r["sample_disco_scores"]) == 4 for r in records)


def test_unsupported_settings_raise() -> None:
    with pytest.raises(ValueError):
        DisCoMBOSampler(num_samples=5)
    with pytest.raises(TypeError):
        DisCoMBOSampler(no_such_argument=1)
    with pytest.raises(TypeError):
        DisCoMBOSampler(exploration_method="stratified")  # deprecated
    with pytest.raises(ValueError):
        DisCoMBOSampler(n_startup_trials=0)
    study = optuna.create_study(
        directions=["minimize", "minimize"], sampler=DisCoMBOSampler(**FAST)
    )
    with pytest.raises(ValueError, match="multi-objective"):
        study.optimize(lambda t: (t.suggest_float("x", 0, 1), t.suggest_float("y", 0, 1)), 12)


def test_pickled_sampler_resumes_the_study() -> None:
    sampler = DisCoMBOSampler(seed=0, **FAST)
    study = optuna.create_study(sampler=sampler)
    study.optimize(mixed_objective, n_trials=14)  # DisCoMBO has proposed a batch

    def resume(sampler: Any) -> list[dict[str, Any]]:
        resumed = optuna.create_study(sampler=sampler)
        resumed.add_trials(study.trials)
        resumed.optimize(mixed_objective, n_trials=6)
        return [t.params for t in resumed.trials[14:]]

    restored = pickle.loads(pickle.dumps(sampler))
    assert resume(restored) == resume(sampler)
