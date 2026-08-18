"""Tests for the bounded, study-local generation retrieval in ``DESampler``.

``DESampler._get_generation_trials`` used to materialize the study's entire trial
history on every generation boundary, which made the per-generation sampling cost grow
with the total number of trials in the study. These tests pin down the optimized
retrieval: that it returns exactly the same trials (and ordering) as the original
full-scan behavior, that it keeps excluding non-``COMPLETE`` trials, that it stays
correct when trial IDs are offset or interleaved by other studies sharing the same
storage, and that it touches only a generation-sized set of trials regardless of how
large the study grows.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import optuna
from optuna.distributions import FloatDistribution
from optuna.storages import BaseStorage
from optuna.storages import InMemoryStorage
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub
import pytest


DESampler = optunahub.load_local_module(
    package="samplers/differential_evolution",
    registry_root="package/",
).DESampler

_GENERATION_KEY = "differential_evolution:generation"


def _full_scan_reference(study: optuna.study.Study, generation: int) -> list[FrozenTrial]:
    """Reference implementation: the original full-history scan and filter."""
    all_trials = study.get_trials(deepcopy=False)
    return [
        t
        for t in all_trials
        if t.state == TrialState.COMPLETE and t.system_attrs.get(_GENERATION_KEY) == generation
    ]


def _objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return x**2 + y**2


def _run_study(
    population_size: int = 8,
    n_trials: int = 40,
    storage: BaseStorage | None = None,
    study_name: str | None = None,
) -> tuple[optuna.study.Study, Any]:
    sampler = DESampler(population_size=population_size, seed=42)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, storage=storage, study_name=study_name
    )
    study.optimize(_objective, n_trials=n_trials)
    return study, sampler


def _observed_generations(study: optuna.study.Study) -> list[int]:
    gens = {
        t.system_attrs[_GENERATION_KEY]
        for t in study.get_trials(deepcopy=False)
        if _GENERATION_KEY in t.system_attrs
    }
    return sorted(gens)


def test_matches_full_scan_including_ordering() -> None:
    # The optimized retrieval must return the same trials, in the same order, as a scan
    # of the entire history for every generation the study produced.
    population_size = 8
    study, sampler = _run_study(population_size=population_size, n_trials=48)

    generations = _observed_generations(study)
    assert len(generations) >= 3  # ensure the study actually spans multiple generations

    for generation in generations:
        expected = _full_scan_reference(study, generation)
        actual = sampler._get_generation_trials(study, generation)

        # Same set and ordering (ordering feeds population-indexed fitness/selection).
        assert [t._trial_id for t in actual] == [t._trial_id for t in expected]
        assert [t.number for t in actual] == [t.number for t in expected]
        assert [t.value for t in actual] == [t.value for t in expected]
        assert all(t.state == TrialState.COMPLETE for t in actual)


def test_excludes_failed_and_pruned_trials() -> None:
    # A generation whose block contains FAIL/PRUNED trials must return only the COMPLETE
    # ones, identically to the full-history scan.
    population_size = 5
    sampler = DESampler(population_size=population_size, seed=0)
    study = optuna.create_study(direction="minimize", sampler=sampler, storage=InMemoryStorage())
    dist = FloatDistribution(-5.0, 5.0)

    # Three full generations (numbers 0..14), with a repeating COMPLETE/FAIL/PRUNED mix
    # so that every generation contains at least one non-COMPLETE trial.
    states = [
        TrialState.COMPLETE,
        TrialState.COMPLETE,
        TrialState.COMPLETE,
        TrialState.FAIL,
        TrialState.PRUNED,
    ]
    for number in range(15):
        state = states[number % len(states)]
        # Fresh InMemory study -> trial_id offset is 0, so number == trial_id here.
        study.add_trial(
            create_trial(
                state=state,
                value=float(number) if state == TrialState.COMPLETE else None,
                params={"x": float(number % 7) - 3.0},
                distributions={"x": dist},
                system_attrs={_GENERATION_KEY: number // population_size},
            )
        )

    for generation in range(3):
        expected = _full_scan_reference(study, generation)
        actual = sampler._get_generation_trials(study, generation)
        assert [t._trial_id for t in actual] == [t._trial_id for t in expected]
        assert all(t.state == TrialState.COMPLETE for t in actual)
        # Sanity: some non-COMPLETE trials really were present in this generation's block.
        assert len(actual) < population_size


def test_correct_with_shared_storage_trial_id_offset() -> None:
    # Trial IDs come from a storage-global counter. When another study occupies the first
    # trial IDs, this study's trials are offset (trial.number != trial._trial_id). The
    # retrieval must still return exactly what the full scan returns.
    population_size = 8
    storage = InMemoryStorage()

    prior = optuna.create_study(storage=storage, study_name="prior")
    prior.optimize(lambda t: t.suggest_float("z", 0.0, 1.0), n_trials=7)

    study, sampler = _run_study(
        population_size=population_size,
        n_trials=40,
        storage=storage,
        study_name="de",
    )

    # Confirm the offset is genuinely non-zero, i.e. the test exercises the offset path.
    offset = storage.get_trial_id_from_study_id_trial_number(study._study_id, 0)
    assert offset != 0

    generations = _observed_generations(study)
    assert len(generations) >= 3
    for generation in generations:
        expected = _full_scan_reference(study, generation)
        actual = sampler._get_generation_trials(study, generation)
        assert [t._trial_id for t in actual] == [t._trial_id for t in expected]
        assert [t.number for t in actual] == [t.number for t in expected]


def test_correct_with_interleaved_shared_storage() -> None:
    # Two studies write trials into one storage in an interleaved order, so a single
    # study's global trial IDs are no longer contiguous (trial.number != trial._trial_id
    # and the gaps are irregular). Because generations are defined on the study-local
    # trial number, retrieval must still match the full scan, and each fully populated
    # generation must contain exactly population_size completed trials.
    population_size = 8
    storage = InMemoryStorage()

    sampler_a = DESampler(population_size=population_size, seed=1)
    sampler_b = DESampler(population_size=population_size, seed=2)
    study_a = optuna.create_study(
        direction="minimize", sampler=sampler_a, storage=storage, study_name="a"
    )
    study_b = optuna.create_study(
        direction="minimize", sampler=sampler_b, storage=storage, study_name="b"
    )

    # Alternate one trial at a time so the two studies' trial IDs interleave in storage.
    for _ in range(5 * population_size):
        study_a.optimize(_objective, n_trials=1)
        study_b.optimize(_objective, n_trials=1)

    # The interleaving must actually make this study's IDs non-contiguous.
    numbers_to_ids = {t.number: t._trial_id for t in study_a.get_trials(deepcopy=False)}
    assert any(numbers_to_ids[n] != n for n in numbers_to_ids)

    generations = _observed_generations(study_a)
    assert len(generations) >= 3
    for generation in generations:
        expected = _full_scan_reference(study_a, generation)
        actual = sampler_a._get_generation_trials(study_a, generation)
        assert [t._trial_id for t in actual] == [t._trial_id for t in expected]
        assert [t.number for t in actual] == [t.number for t in expected]

    # A later, fully populated generation must hold exactly population_size trials, i.e.
    # generation bookkeeping stays well-formed under interleaving.
    assert len(sampler_a._get_generation_trials(study_a, generations[-2])) == population_size


def test_retrieval_touches_only_generation_sized_set_independent_of_history() -> None:
    # Regression guard for the scaling fix, without wall-clock assertions: the number of
    # per-trial storage fetches for one generation must be bounded by population_size and
    # must not grow as the study accumulates more generations.
    population_size = 8
    sampler = DESampler(population_size=population_size, seed=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, storage=InMemoryStorage())

    def count_get_trial_calls(target_generation: int) -> int:
        storage = study._storage
        with patch.object(storage, "get_trial", wraps=storage.get_trial) as spy_get_trial:
            with patch.object(
                storage, "get_all_trials", wraps=storage.get_all_trials
            ) as spy_get_all:
                sampler._get_generation_trials(study, target_generation)
        # The full-history path must not be used at all.
        assert spy_get_all.call_count == 0
        return int(spy_get_trial.call_count)

    # Grow the study and measure the cost of retrieving the SAME early generation as the
    # total number of trials increases. A history-scanning implementation would touch more
    # trials as the study grows; the bounded implementation must not.
    study.optimize(_objective, n_trials=3 * population_size)
    calls_small = count_get_trial_calls(target_generation=1)

    study.optimize(_objective, n_trials=9 * population_size)
    calls_large = count_get_trial_calls(target_generation=1)

    assert calls_small <= population_size
    assert calls_large <= population_size
    # The essential scaling property: retrieval cost is independent of the study size.
    assert calls_small == calls_large


def test_generation_semantics_slot_gaps_and_no_completion_barrier() -> None:
    # Makes the generation invariant explicit (as requested in review):
    # (1) FAIL/PRUNED trials consume their trial-number slot, so a generation can
    #     hold fewer than population_size completed trials; and
    # (2) generation boundaries are not completion barriers -- a later generation
    #     can be fully present while an earlier one is left under-filled.
    population_size = 5
    sampler = DESampler(population_size=population_size, seed=0)
    study = optuna.create_study(direction="minimize", sampler=sampler, storage=InMemoryStorage())
    dist = FloatDistribution(-5.0, 5.0)

    # Generation 0 (numbers 0..4): #2 FAIL, #3 PRUNED -> only 3 completed.
    # Generation 1 (numbers 5..9): all completed.
    non_complete = {2: TrialState.FAIL, 3: TrialState.PRUNED}
    for number in range(10):
        state = non_complete.get(number, TrialState.COMPLETE)
        study.add_trial(
            create_trial(
                state=state,
                value=float(number) if state == TrialState.COMPLETE else None,
                params={"x": float(number % 7) - 3.0},
                distributions={"x": dist},
                system_attrs={_GENERATION_KEY: number // population_size},
            )
        )

    gen0 = sampler._get_generation_trials(study, 0)
    gen1 = sampler._get_generation_trials(study, 1)

    # (1) Under-filled generation: fewer than population_size completed trials.
    assert [t.number for t in gen0] == [0, 1, 4]
    assert len(gen0) < population_size

    # (2) No completion barrier: the later generation is fully present even though
    #     the earlier one never reached population_size completed trials.
    assert [t.number for t in gen1] == [5, 6, 7, 8, 9]
    assert len(gen1) == population_size


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
