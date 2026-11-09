from __future__ import annotations

from typing import Any

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState
import optunahub
import pytest


MetaLearnTPESampler = optunahub.load_local_module(
    package="samplers/meta_learn_tpe", registry_root="package/"
).MetaLearnTPESampler
_GROUPS = {frozenset(["branch", "common"]), frozenset(["y", "y2"]), frozenset(["z"])}


def _add_trials(study: optuna.Study, branches: tuple[str, ...] = ("left", "right")) -> None:
    for branch in branches:
        for i in range(10):
            value = (i + 1) / 11
            params: dict[str, Any] = {"branch": branch, "common": value, "fixed": 1.0}
            distributions: dict[str, Any] = {
                "branch": CategoricalDistribution(["left", "right"]),
                "common": FloatDistribution(0, 1),
                "fixed": FloatDistribution(1, 1),
            }
            conditional = {"y": value, "y2": 1 - value} if branch == "left" else {"z": value}
            params.update(conditional)
            distributions.update({name: FloatDistribution(0, 1) for name in conditional})
            study.add_trial(
                optuna.trial.create_trial(
                    params=params,
                    distributions=distributions,
                    values=[value] if len(study.directions) == 1 else [value, 1 - value],
                )
            )


def _sources(directions: list[str]) -> list[optuna.Study]:
    # The first source has no usable observations; the other sources cover disjoint branches.
    empty = optuna.create_study(directions=directions)
    left = optuna.create_study(directions=directions)
    right = optuna.create_study(directions=directions)
    _add_trials(left, ("left",))
    _add_trials(right, ("right",))
    return [empty, left, right]


def _objective(trial: optuna.Trial) -> float:
    branch = trial.suggest_categorical("branch", ["left", "right"])
    value = trial.suggest_float("common", 0, 1)
    if branch == "left":
        value += trial.suggest_float("y", 0, 1) + trial.suggest_float("y2", 0, 1)
    else:
        value += trial.suggest_float("z", 0, 1)
    trial.suggest_float("fixed", 1, 1)
    return value


@pytest.mark.parametrize("constant_liar", [False, True])
@pytest.mark.parametrize("n_objectives", [1, 2])
def test_conditional_groups_use_matching_observations(
    monkeypatch: pytest.MonkeyPatch, constant_liar: bool, n_objectives: int
) -> None:
    directions = ["minimize"] * n_objectives
    sampler = MetaLearnTPESampler(
        source_studies=_sources(directions),
        n_startup_trials=5,
        seed=0,
        group=True,
        constant_liar=constant_liar,
    )
    study = optuna.create_study(directions=directions, sampler=sampler)
    _add_trials(study)
    observed: dict[tuple[int, frozenset[str], bool], list[optuna.trial.FrozenTrial]] = {}

    def record_model(index: int, internal_sampler: optuna.samplers.TPESampler) -> None:
        build = internal_sampler._build_parzen_estimator

        def capture(
            study: optuna.Study,
            search_space: dict[str, Any],
            trials: list[optuna.trial.FrozenTrial],
            handle_below: bool,
        ) -> Any:
            assert trials
            assert all(search_space.keys() <= trial.params.keys() for trial in trials)
            observed[index, frozenset(search_space), handle_below] = trials
            return build(study, search_space, trials, handle_below)

        monkeypatch.setattr(internal_sampler, "_build_parzen_estimator", capture)

    record_model(-1, sampler._target_sampler)
    for index, internal_sampler in enumerate(sampler._source_samplers):
        record_model(index, internal_sampler)
    trial = study.ask()
    trial.suggest_categorical("branch", ["left", "right"])
    assert set(trial.relative_params) == {"branch", "common", "y", "y2", "z"}
    assert {keys for index, keys, _ in observed if index == -1} == _GROUPS
    for keys in _GROUPS:
        expected = 20 if "branch" in keys else 10
        assert len(observed[-1, keys, True]) + len(observed[-1, keys, False]) == expected
    assert not any(index == 0 for index, _, _ in observed)
    assert not any(index == 1 and "z" in keys for index, keys, _ in observed)
    assert not any(index == 2 and "y" in keys for index, keys, _ in observed)
    value = _objective(trial)
    study.tell(trial, value if n_objectives == 1 else (value, -value))

    def objective(trial: optuna.Trial) -> float | tuple[float, float]:
        value = _objective(trial)
        return value if n_objectives == 1 else (value, -value)

    study.optimize(objective, n_trials=6)
    assert all(trial.state == TrialState.COMPLETE for trial in study.trials)


@pytest.mark.parametrize("n_startup_trials", [5, 15])
@pytest.mark.parametrize("matching_source", [False, True])
def test_fallback_respects_requested_subspace(
    n_startup_trials: int, matching_source: bool
) -> None:
    source = optuna.create_study()
    if matching_source:
        _add_trials(source)
    else:
        source.add_trial(
            optuna.trial.create_trial(
                params={"unrelated": 0.5},
                distributions={"unrelated": FloatDistribution(0, 1)},
                value=1.0,
            )
        )
    sampler = MetaLearnTPESampler(
        source_studies=[source], n_startup_trials=n_startup_trials, group=True, seed=0
    )
    study = optuna.create_study(sampler=sampler)
    _add_trials(study)
    study.ask()
    frozen = study.trials[-1]
    sampler.infer_relative_search_space(study, frozen)
    params = sampler.sample_relative(study, frozen, {"y2": FloatDistribution(0, 1)})
    assert set(params) == {"y2"}
    assert 0 <= params["y2"] <= 1


def test_partial_fixed_sampler() -> None:
    base = MetaLearnTPESampler(source_studies=_sources(["minimize"]), group=True, seed=0)
    fixed = {"branch": "left", "common": 0.25, "y": 0.9}
    study = optuna.create_study(sampler=optuna.samplers.PartialFixedSampler(fixed, base))
    _add_trials(study)
    trial = study.ask()
    _objective(trial)
    assert set(trial.relative_params) == {"y2", "z"}
    assert all(trial.params[name] == value for name, value in fixed.items())


def test_group_with_constant_liar_shares_all_subspaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_studies = _sources(["minimize"])
    sampler = MetaLearnTPESampler(
        source_studies=source_studies, n_startup_trials=5, group=True, constant_liar=True, seed=0
    )
    study = optuna.create_study(sampler=sampler)
    _add_trials(study)
    pending = study.ask()
    pending.suggest_categorical("branch", ["left", "right"])
    assert set(pending.params) == {"branch"}
    worker_sampler = MetaLearnTPESampler(
        source_studies=source_studies, n_startup_trials=5, group=True, constant_liar=True, seed=1
    )
    worker = optuna.load_study(
        study_name=study.study_name, storage=study._storage, sampler=worker_sampler
    )
    seen: set[frozenset[str]] = set()
    build = worker_sampler._target_sampler._build_parzen_estimator

    def capture(
        study: optuna.Study,
        search_space: dict[str, Any],
        trials: list[optuna.trial.FrozenTrial],
        handle_below: bool,
    ) -> Any:
        if not handle_below:
            previous = next(trial for trial in trials if trial.number == pending.number)
            assert all(
                previous.params[name] == pending.relative_params[name] for name in search_space
            )
            seen.add(frozenset(search_space))
        return build(study, search_space, trials, handle_below)

    monkeypatch.setattr(worker_sampler._target_sampler, "_build_parzen_estimator", capture)
    worker.ask().suggest_categorical("branch", ["left", "right"])
    assert seen == _GROUPS
    assert set(study.trials[pending.number].params) == {"branch"}


def test_single_valued_search_space() -> None:
    source = optuna.create_study()
    distribution = FloatDistribution(1, 1)
    source.add_trial(
        optuna.trial.create_trial(
            params={"fixed": 1.0}, distributions={"fixed": distribution}, value=1
        )
    )
    study = optuna.create_study(
        sampler=MetaLearnTPESampler(source_studies=[source], group=True, seed=0)
    )
    study.add_trials(source.trials)
    trial = study.ask()
    assert trial.suggest_float("fixed", 1, 1) == 1
    assert trial.relative_params == {}


@pytest.mark.parametrize("constant_liar", [False, True])
def test_subspace_without_completed_target_observations(constant_liar: bool) -> None:
    sampler = MetaLearnTPESampler(
        source_studies=_sources(["minimize"]),
        group=True,
        constant_liar=constant_liar,
        n_startup_trials=0,
        seed=0,
    )
    study = optuna.create_study(sampler=sampler)
    _add_trials(study, ("left",))
    # The right branch is only known from a pruned trial.
    study.add_trial(
        optuna.trial.create_trial(
            state=TrialState.PRUNED,
            params={"branch": "right", "common": 0.5, "z": 0.6},
            distributions={
                "branch": CategoricalDistribution(["left", "right"]),
                "common": FloatDistribution(0, 1),
                "z": FloatDistribution(0, 1),
            },
            intermediate_values={0: 0.5},
        )
    )
    trial = study.ask()
    trial.suggest_categorical("branch", ["left", "right"])
    assert 0 <= trial.relative_params["z"] <= 1
