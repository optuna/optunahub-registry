# mypy: ignore-errors
"""OptQuestSampler tests, run against the engine installed by `pip install optquest`.

Every test builds its own sampler (its own engine instance) and asks for fewer than 50 trials,
so the suite runs in demo mode and needs no license. The engine is deterministic for a given
model and seed, so the assertions do not depend on the search path.

Run from the root of the registry: `pytest package/samplers/optquest/tests`
"""

import ast
from collections import Counter
import decimal
import importlib
import math
import os
import re
import threading
import warnings

import optuna
import optunahub
import pytest


try:
    import optquest

    optquest.get_version_info()
except Exception as e:  # ImportError, or OSError when the engine library cannot be loaded
    pytest.skip(
        f"the OptQuest engine is unavailable on this platform ({e})", allow_module_level=True
    )


module = optunahub.load_local_module(package="samplers/optquest", registry_root="package/")
OptQuestSampler = module.OptQuestSampler
_package = OptQuestSampler.__module__.rsplit(".", 1)[0]
_variable_codec = importlib.import_module(f"{_package}._variable_codec")
OptQuestOutputMismatchError = importlib.import_module(
    f"{_package}.OptQuestSampler"
).OptQuestOutputMismatchError

optuna.logging.set_verbosity(optuna.logging.ERROR)

D = optuna.distributions
MIN = optuna.study.StudyDirection.MINIMIZE
MAX = optuna.study.StudyDirection.MAXIMIZE

SPHERE_SPACE = {"x": D.FloatDistribution(-5, 10), "y": D.FloatDistribution(-5, 10)}


def sphere(trial):
    x = trial.suggest_float("x", -5, 10)
    y = trial.suggest_float("y", -5, 10)
    return x * x + y * y


def _optimize(study, objective, n_trials, **kwargs):
    """Run the study and return the warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study.optimize(objective, n_trials=n_trials, **kwargs)
    return [str(w.message) for w in caught]


def _states(study):
    return Counter(t.state.name for t in study.trials)


def _sphere_sampler(**kwargs):
    kwargs.setdefault("seed", 3)
    return OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN], license="", **kwargs)


# ---------------------------------------------------------------------------------------------
# Distribution mapping
# ---------------------------------------------------------------------------------------------

MIXED_SPACE = {
    "f": D.FloatDistribution(-5, 10),
    "flog": D.FloatDistribution(1e-3, 10.0, log=True),
    "fstep": D.FloatDistribution(-5, 10, step=0.5),
    "i": D.IntDistribution(1, 10),
    "ilog": D.IntDistribution(1, 1000, log=True),
    "istep": D.IntDistribution(1, 10, step=3),
    "c": D.CategoricalDistribution(["a", "b", "c"]),
    "cnum": D.CategoricalDistribution([1, 2.5, None]),
}


def mixed_objective(trial):
    f = trial.suggest_float("f", -5, 10)
    flog = trial.suggest_float("flog", 1e-3, 10.0, log=True)
    fstep = trial.suggest_float("fstep", -5, 10, step=0.5)
    i = trial.suggest_int("i", 1, 10)
    ilog = trial.suggest_int("ilog", 1, 1000, log=True)
    istep = trial.suggest_int("istep", 1, 10, step=3)
    c = trial.suggest_categorical("c", ["a", "b", "c"])
    cnum = trial.suggest_categorical("cnum", [1, 2.5, None])
    return (
        f * f
        + (math.log10(flog) + 1) ** 2
        + fstep**2
        + i
        + math.log(ilog)
        + istep
        + {"a": 0, "b": 1, "c": 2}[c]
        + (0 if cnum is None else cnum)
    )


def test_search_space_distributions_declare_the_engine_variables():
    sampler = OptQuestSampler(search_space=MIXED_SPACE, directions=[MIN], license="", seed=7)
    lines = {
        line.split(" ")[1]: line for line in sampler._model.describe().splitlines() if " " in line
    }
    assert lines["f"] == "CONTINUOUS f -5 10"
    assert lines["flog"].startswith("CONTINUOUS flog -6.907755") and lines["flog"].endswith(
        "2.3025850929940459"
    )
    assert lines["fstep"] == "DISCRETE fstep -5 10 0.5"
    assert lines["i"] == "INTEGER i 1 10"
    assert lines["ilog"].startswith(
        "CONTINUOUS ilog 0 6.907755"
    )  # log-space CONTINUOUS, not INTEGER
    assert lines["istep"] == "DISCRETE istep 1 10 3"
    assert lines["c"] == "DESIGN c 3"
    assert lines["cnum"] == "DESIGN cnum 3"
    assert lines["OQobj00"] == "OBJECTIVE OQobj00 MIN OQout00"
    assert "RANDOMSEED 7" in sampler._model.describe()
    assert sampler.directions == [MIN]


def test_engine_proposals_are_taken_as_proposed_never_resampled():
    """Every value the engine proposes lands inside its Optuna distribution (log bounds clamped,
    discrete grids exact, categorical indices decoded), so Optuna never falls back to
    sample_independent, and the engine's record of each trial equals trial.params."""
    sampler = OptQuestSampler(search_space=MIXED_SPACE, directions=[MIN], license="", seed=7)
    fallbacks = []
    original = sampler.sample_independent

    def counting(study, trial, name, dist):
        fallbacks.append(name)
        return original(study, trial, name, dist)

    sampler.sample_independent = counting
    study = optuna.create_study(sampler=sampler, directions=[MIN])
    assert _optimize(study, mixed_objective, 30) == []
    assert fallbacks == []
    solutions = sorted(sampler._model.get_all_solutions(), key=lambda s: s.get_iteration())
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(solutions) == len(trials) == 30
    for sol, trial in zip(solutions, trials):
        p = trial.params
        assert sol.get_value("f") == p["f"]
        assert math.isclose(math.exp(sol.get_value("flog")), p["flog"], rel_tol=1e-9)
        assert sol.get_value("fstep") == p["fstep"]
        assert sol.get_value("i") == p["i"]
        assert round(math.exp(sol.get_value("ilog"))) == p["ilog"]
        assert sol.get_value("istep") == p["istep"]
        assert ["a", "b", "c"][int(sol.get_value("c")) - 1] == p["c"]
        assert [1, 2.5, None][int(sol.get_value("cnum")) - 1] == p["cnum"]
        assert p["istep"] in (1, 4, 7, 10)
        assert isinstance(p["i"], int) and isinstance(p["ilog"], int)


def _sphere_points(seed, n_trials=16):
    """The points of a sphere run. The engine's first 11 proposals do not depend on the seed, so
    a run must be longer than that to show the seed."""
    sampler = OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN], license="", seed=seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(sphere, n_trials=n_trials)
    return [(t.params["x"], t.params["y"]) for t in study.trials]


@pytest.mark.parametrize(
    "seed, engine_seed", [(11, 11), (1, 1), (0, 2**31 - 1), (-5, 2**31 - 6), (2**40, 512)]
)
def test_every_int_seed_reaches_the_engine(seed, engine_seed):
    """The engine takes a seed from 1 to 2**31 - 1 and ignores any other value, so the sampler
    gives it (seed - 1) mod (2**31 - 1) + 1: seed=0, a negative seed and a large seed all
    apply, and the seeds of a run over range(n) are all different."""
    sampler = OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN], license="", seed=seed)
    assert f"RANDOMSEED {engine_seed}" in sampler._model.describe().splitlines()


def test_a_seed_changes_the_search_and_repeats_it():
    unseeded = _sphere_points(None)
    assert _sphere_points(None) == unseeded
    runs = {seed: _sphere_points(seed) for seed in (0, 1, 2, -5)}
    assert runs[0] == _sphere_points(0)
    for seed, points in runs.items():
        assert points != unseeded, seed
    assert len({tuple(points) for points in runs.values()}) == len(runs)  # 0 and 1 differ
    with pytest.raises(TypeError, match="seed must be an int"):
        OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN], license="", seed=1.5)


def test_model_mode_refuses_a_seed_it_cannot_apply():
    """set_random_seed answers False on a model the caller already initialized (ERROR already
    initialized); the sampler raises instead of running unseeded."""
    model = optquest.OptQuestModel()
    model.set_license("")
    model.add_continuous_variable("x", -5, 5)
    model.add_output_variable("out")
    model.add_minimize_objective("obj", "out")
    assert model.initialize()
    with pytest.raises(RuntimeError, match="seed=7 was not applied"):
        OptQuestSampler(model=model, seed=7)


CONSTRAINED = [
    {"cmd": "CONTINUOUS", "name": "x", "min": -5.0, "max": 10.0},
    {"cmd": "CONTINUOUS", "name": "y", "min": -5.0, "max": 10.0},
    {"cmd": "OUTPUT", "name": "score"},
    {"cmd": "OUTPUT", "name": "budget"},
    {"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "score"},
    {"cmd": "CONSTRAINT", "name": "lim", "expression": "budget <= 3"},
]


# ---------------------------------------------------------------------------------------------
# Trial states reaching the engine
# ---------------------------------------------------------------------------------------------


def test_unregistered_parameter_is_skipped_and_warned_once():
    """A parameter the objective suggests outside the search space has no engine variable;
    it is left out of the engine's record (a write would fail submit() with ERROR BAD VALUE
    NAME) and warned about once, and every trial is still recorded."""

    def objective(trial):
        z = trial.suggest_int("z", 0, 3)
        mode = trial.suggest_categorical("mode", ["fast", "slow"])
        return sphere(trial) + z + (1 if mode == "slow" else 0)

    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    caught = _optimize(study, objective, 6)
    assert len(caught) == 2 and all("not in this sampler's search space" in w for w in caught)
    assert sorted(w.split("'")[1] for w in caught) == ["mode", "z"]
    assert sampler._model.get_number_of_completed_iterations() == 6
    assert len(sampler._model.get_all_solutions()) == 6
    assert _states(study) == {"COMPLETE": 6}


def test_pruned_and_failed_trials_are_rejected():
    def objective(trial):
        value = sphere(trial)
        if trial.number == 2:
            raise optuna.TrialPruned()
        if trial.number == 3:
            raise ValueError("boom")
        return value

    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    assert _optimize(study, objective, 8, catch=(ValueError,)) == []
    assert [t.state.name for t in study.trials] == ["COMPLETE", "COMPLETE", "PRUNED", "FAIL"] + [
        "COMPLETE"
    ] * 4
    assert (
        sampler._model.get_number_of_completed_iterations() == 8
    )  # a reject completes an iteration
    assert sampler._solutions == {}


def test_an_uncaught_exception_rejects_its_trial_and_the_study_goes_on():
    def objective(trial):
        value = sphere(trial)
        if trial.number == 1:
            raise ValueError("boom")
        return value

    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    with pytest.raises(ValueError, match="boom"):
        study.optimize(objective, n_trials=4)
    assert [t.state.name for t in study.trials] == ["COMPLETE", "FAIL"]
    assert sampler._solutions == {}
    assert _optimize(study, sphere, 3) == []
    assert sampler._model.get_number_of_completed_iterations() == 5


def test_nan_output_rejects_the_trial():
    """A NaN in a non-objective output (Optuna itself fails a trial whose objective value is
    NaN) is refused: the trial stays COMPLETE for Optuna and is rejected in the engine."""
    sampler = OptQuestSampler(statements=CONSTRAINED, license="", seed=5)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    caught = _optimize(
        study, sampler.wrap(lambda p: {"score": p["x"] ** 2, "budget": float("nan")}), 3
    )
    assert len(caught) == 3 and all("cannot accept (budget=nan)" in w for w in caught)
    assert sampler._model.get_number_of_completed_iterations() == 3
    assert _states(study) == {"COMPLETE": 3}


def test_demo_cap_ends_the_study_cleanly():
    sampler = OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN], license="", seed=3)
    study = optuna.create_study(sampler=sampler)
    caught = _optimize(study, sphere, 60)
    assert _states(study) == {"COMPLETE": 50, "PRUNED": 1}
    assert sampler._model.get_termination_reason() == optquest.OQTerminationReasonMaxDemoIterations
    assert len(caught) == 1 and "demo-license cap" in caught[0]


def test_finite_space_exhaustion_is_a_clean_end():
    space = {"c": D.CategoricalDistribution(["a", "b", "c"]), "b": D.IntDistribution(0, 1)}
    sampler = OptQuestSampler(search_space=space, directions=[MIN], license="", seed=3)
    study = optuna.create_study(sampler=sampler)

    def objective(trial):
        return {"a": 0, "b": 1, "c": 2}[
            trial.suggest_categorical("c", ["a", "b", "c"])
        ] + trial.suggest_int("b", 0, 1)

    assert _optimize(study, objective, 12) == []
    assert _states(study) == {"COMPLETE": 6, "PRUNED": 1}
    assert sampler._model.get_termination_reason() == optquest.OQTerminationReasonOptimalFound


def test_n_jobs_serializes_the_engine_handshake():
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    assert _optimize(study, sphere, 30, n_jobs=3) == []
    assert _states(study) == {"COMPLETE": 30}
    assert sampler._model.get_number_of_completed_iterations() == 30
    assert sampler._solutions == {}


# ---------------------------------------------------------------------------------------------
# Externally evaluated points: enqueue_trial and a study with history
# ---------------------------------------------------------------------------------------------


def test_enqueued_point_is_added_as_evaluated_and_reissued_once():
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    study.enqueue_trial({"x": 2.0, "y": 3.0})
    assert _optimize(study, sphere, 8) == []
    assert _states(study) == {"COMPLETE": 8}
    assert study.trials[0].params == {"x": 2.0, "y": 3.0} and study.trials[0].value == 13.0
    solutions = sampler._model.get_all_solutions()
    assert len(solutions) == 8 and sampler._model.get_number_of_completed_iterations() == 8
    enqueued = [s for s in solutions if s.get_value("x") == 2.0 and s.get_value("y") == 3.0]
    assert (
        len(enqueued) == 1
        and enqueued[0].is_evaluated()
        and enqueued[0].get_value("OQout00") == 13.0
    )
    assert sum(1 for s in solutions if s.is_evaluated()) == 1


def test_partially_enqueued_point_takes_its_free_dimension_from_sample_independent():
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    study.enqueue_trial({"x": 1.0})
    assert _optimize(study, sphere, 4) == []
    assert study.trials[0].params["x"] == 1.0 and -5 <= study.trials[0].params["y"] <= 10
    assert len(sampler._model.get_all_solutions()) == 4


def test_attaching_to_a_study_with_history_feeds_the_history_to_the_engine():
    study = optuna.create_study()
    study.optimize(sphere, n_trials=5)
    sampler = _sphere_sampler()
    study.sampler = sampler
    assert _optimize(study, sphere, 5) == []
    assert len(study.trials) == 10
    assert len(sampler._model.get_all_solutions()) == 10
    assert sum(1 for s in sampler._model.get_all_solutions() if s.is_evaluated()) == 5


# ---------------------------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------------------------


def test_multi_objective_directions_and_values_reach_the_engine():
    sampler = OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN, MAX], license="", seed=3)
    assert sampler.directions == [MIN, MAX]
    study = optuna.create_study(sampler=sampler, directions=["minimize", "maximize"])

    def objective(trial):
        x = trial.suggest_float("x", -5, 10)
        y = trial.suggest_float("y", -5, 10)
        return x * x, y

    assert _optimize(study, objective, 20) == []
    described = sampler._model.describe()
    assert (
        "OBJECTIVE OQobj00 MIN OQout00" in described
        and "OBJECTIVE OQobj01 MAX OQout01" in described
    )
    for sol in sampler._model.get_all_solutions():
        assert sol.get_value("OQout01") == sol.get_value("y")
        assert math.isclose(sol.get_value("OQout00"), sol.get_value("x") ** 2, rel_tol=1e-9)


def test_study_directions_must_match_the_declared_objectives():
    sampler = OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN, MAX], license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize"])
    with pytest.raises(ValueError, match="direction\\[1\\] is MINIMIZE"):
        study.optimize(
            lambda t: (t.suggest_float("x", -5, 10), t.suggest_float("y", -5, 10)), n_trials=1
        )


# ---------------------------------------------------------------------------------------------
# statements= and model= construction, wrap()
# ---------------------------------------------------------------------------------------------


def test_wrap_records_every_declared_output_and_rejects_a_none_result():
    sampler = OptQuestSampler(statements=CONSTRAINED, license="", seed=5)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)

    def sim(p):
        if p["x"] < -4.5:
            return None
        return {"score": p["x"] ** 2 + p["y"] ** 2, "budget": p["x"] + p["y"]}

    assert _optimize(study, sampler.wrap(sim), 15) == []
    states = _states(study)
    assert states["COMPLETE"] + states.get("PRUNED", 0) == 15
    solutions = sampler._model.get_all_solutions()
    assert len(solutions) == 15
    submitted = [s for s in solutions if s.get_value("x") >= -4.5]
    assert submitted and all(
        math.isclose(s.get_value("budget"), s.get_value("x") + s.get_value("y"), abs_tol=1e-9)
        for s in submitted
    )
    assert study.best_trial.value == min(t.value for t in study.trials if t.value is not None)


def test_wrap_refuses_an_output_dict_that_does_not_match_the_declaration():
    sampler = OptQuestSampler(statements=CONSTRAINED, license="", seed=5)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    with pytest.raises(OptQuestOutputMismatchError, match="missing=\\['budget'\\]"):
        study.optimize(sampler.wrap(lambda p: {"score": 1.0}), n_trials=1)


def test_manual_objective_with_an_unlinked_output_raises_at_the_first_trial():
    """A hand-written objective has no value for an output that no objective passes through, so
    the engine would mark every trial infeasible, and every trial would be rejected. The sampler
    rejects the first trial in the engine and raises, instead of running a search that gets no
    result."""
    sampler = OptQuestSampler(statements=CONSTRAINED, license="", seed=5)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    with pytest.raises(ValueError, match=r"output\(s\) \['budget'\].*sampler.wrap\(sim\)"):
        study.optimize(
            lambda t: t.suggest_float("x", -5, 10) ** 2 + t.suggest_float("y", -5, 10) ** 2,
            n_trials=3,
        )
    assert _states(study) == {"COMPLETE": 1}
    assert sampler._solutions == {}
    assert sampler._model.get_number_of_completed_iterations() == 1  # the trial was rejected


def test_earlier_trials_without_every_output_are_not_sent_and_warned_once():
    """Trials that are in the study before the sampler attaches cannot be stopped: the sampler
    does not send the ones without a value for every output, and warns once."""
    study = optuna.create_study()
    study.optimize(
        lambda t: t.suggest_float("x", -5, 10) ** 2 + t.suggest_float("y", -5, 10) ** 2,
        n_trials=3,
    )
    sampler = OptQuestSampler(statements=CONSTRAINED, license="", seed=5)
    study.sampler = sampler

    def sim(p):
        return {"score": p["x"] ** 2 + p["y"] ** 2, "budget": p["x"] + p["y"]}

    caught = _optimize(study, sampler.wrap(sim), 2)
    assert len(caught) == 1 and "in the study before this sampler" in caught[0]
    assert len(study.trials) == 5
    assert len(sampler._model.get_all_solutions()) == 2  # the earlier trials were not sent


def test_non_identity_objective_is_refused_at_construction():
    statements = [
        {"cmd": "CONTINUOUS", "name": "x", "min": -5.0, "max": 10.0},
        {"cmd": "OUTPUT", "name": "score"},
        {"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "score * 2"},
    ]
    with pytest.raises(NotImplementedError, match="identity pass-through"):
        OptQuestSampler(statements=statements, license="")


def test_typed_model_mode_recovers_the_search_space_from_describe():
    model = optquest.OptQuestModel()
    model.set_license("")
    model.add_continuous_variable("x", -5, 10)
    model.add_integer_variable("k", 1, 5)
    model.add_design_variable("d", 3)
    model.add_permutation_variable("perm", ["a", "b", "c"])
    model.add_mixture_variable("mix", ["p", "q"], None, None, 1.0)
    model.add_output_variable("out")
    model.add_minimize_objective("obj", "out")
    sampler = OptQuestSampler(model=model, seed=2)
    assert {k: type(v).__name__ for k, v in sampler._space.items()} == {
        "x": "FloatDistribution",
        "k": "IntDistribution",
        "d": "CategoricalDistribution",
        "perm@a": "IntDistribution",
        "perm@b": "IntDistribution",
        "perm@c": "IntDistribution",
        "mix@p": "FloatDistribution",
        "mix@q": "FloatDistribution",
    }
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    seen = []

    def sim(p):
        seen.append(p)
        return {"out": p["x"] ** 2 + p["k"] + int(p["d"]) + p["perm"]["a"] + p["mix"]["p"]}

    assert _optimize(study, sampler.wrap(sim), 8) == []
    assert _states(study) == {"COMPLETE": 8}
    for p in seen:
        assert sorted(p["perm"].values()) == [1, 2, 3]
        assert math.isclose(p["mix"]["p"] + p["mix"]["q"], 1.0, abs_tol=1e-9)
        assert p["d"] in (1, 2, 3)


# ---------------------------------------------------------------------------------------------
# Codec shapes with engine-specific value conventions
# ---------------------------------------------------------------------------------------------


def test_selection_is_a_binary_gate():
    statements = [
        {"cmd": "CONTINUOUS", "name": "s1", "min": 0.0, "max": 1.0},
        {"cmd": "CONTINUOUS", "name": "s2", "min": 0.0, "max": 1.0},
        {"cmd": "SELECTION", "name": "sel", "vars": ["s1", "s2"]},
        {"cmd": "OUTPUT", "name": "out"},
        {"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "out"},
    ]
    sampler = OptQuestSampler(statements=statements, license="", seed=4)
    assert sampler._space["sel"] == D.CategoricalDistribution([False, True])
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    seen = []

    def sim(p):
        seen.append(p["sel"])
        return {"out": p["s1"] + p["s2"]}

    assert _optimize(study, sampler.wrap(sim), 10) == []
    assert all(value in ([], ["s1", "s2"]) for value in seen)
    assert {t.params["sel"] for t in study.trials} <= {False, True}
    codec = sampler._var_codecs["sel"]
    assert codec.to_engine["sel"](True) == 1.0 and codec.to_engine["sel"](False) == 0.0
    with pytest.raises(ValueError, match="binary gate"):
        codec.from_engine["sel"](2.0)


def test_angle_azimuth_is_folded_into_the_declared_window():
    statements = [
        {"cmd": "ANGLE", "name": "dir", "min_deg": -45.0, "max_deg": 45.0},
        {
            "cmd": "SPHERICALANGLE",
            "name": "sph",
            "min_theta_deg": -80.0,
            "max_theta_deg": 80.0,
            "min_phi_deg": -45.0,
            "max_phi_deg": 45.0,
        },
        {"cmd": "OUTPUT", "name": "out"},
        {"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "out"},
    ]
    sampler = OptQuestSampler(statements=statements, license="", seed=4)
    fallbacks = []
    original = sampler.sample_independent

    def counting(study, trial, name, dist):
        fallbacks.append(name)
        return original(study, trial, name, dist)

    sampler.sample_independent = counting
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert (
        _optimize(study, sampler.wrap(lambda p: {"out": abs(p["dir"]) + abs(p["sph"]["phi"])}), 10)
        == []
    )
    assert fallbacks == []
    for t in study.trials:
        assert -45.0 <= t.params["dir@theta"] <= 45.0
        assert -80.0 <= t.params["sph@theta"] <= 80.0 and -45.0 <= t.params["sph@phi"] <= 45.0
    fold = sampler._var_codecs["dir"].from_engine["dir@theta"]
    assert (
        fold(315.0) == -45.0
        and math.isclose(fold(338.95), -21.05)
        and math.isclose(fold(35.38), 35.38)
    )
    full = _variable_codec.build(
        {"cmd": "ANGLE", "name": "full", "min_deg": 0.0, "max_deg": 360.0}
    )
    assert full.from_engine["full@theta"](315.0) == 315.0


def _run_counting_fallbacks(sampler, objective, n_trials, enqueue=None):
    """Run a study and return (study, warnings, the params that Optuna sampled at random
    because it rejected the engine's value)."""
    fallbacks = []
    original = sampler.sample_independent

    def counting(study, trial, name, dist):
        fallbacks.append(name)
        return original(study, trial, name, dist)

    sampler.sample_independent = counting
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    if enqueue is not None:
        study.enqueue_trial(enqueue)
    caught = _optimize(study, sampler.wrap(objective), n_trials)
    return study, caught, fallbacks


def _with_output(*variables, direction="min"):
    return [
        *variables,
        {"cmd": "OUTPUT", "name": "out"},
        {"cmd": "OBJECTIVE", "name": "obj", "direction": direction, "expression": "out"},
    ]


def test_a_full_circle_that_does_not_start_at_zero_keeps_the_engine_values():
    """The engine reports an angle in [0, 360). On a full circle declared from -180 to 180, the
    codec moves the engine's 315 to -45, so that Optuna takes the engine's value instead of
    sampling a random one."""
    statements = _with_output(
        {"cmd": "ANGLE", "name": "dir", "min_deg": -180.0, "max_deg": 180.0},
        {
            "cmd": "SPHERICALANGLE",
            "name": "sph",
            "min_theta_deg": -80.0,
            "max_theta_deg": 80.0,
            "min_phi_deg": 90.0,
            "max_phi_deg": 450.0,
        },
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=4)
    study, caught, fallbacks = _run_counting_fallbacks(
        sampler, lambda p: {"out": abs(p["dir"]) + abs(p["sph"]["phi"] - 270.0)}, 30
    )
    assert caught == [] and fallbacks == []
    for t in study.trials:
        assert -180.0 <= t.params["dir@theta"] <= 180.0
        assert 90.0 <= t.params["sph@phi"] <= 450.0
    fold = sampler._var_codecs["dir"].from_engine["dir@theta"]
    assert fold(315.0) == -45.0 and fold(0.0) == 0.0 and fold(180.0) == -180.0
    fold_phi = sampler._var_codecs["sph"].from_engine["sph@phi"]
    assert fold_phi(10.0) == 370.0 and fold_phi(90.0) == 90.0


def test_an_arc_with_min_above_max_crosses_zero_and_is_registered_above_360():
    """An arc from 320 to 40 goes through 0. Optuna cannot take FloatDistribution(320, 40), so
    the codec registers 320 to 400, and 10 degrees is the value 370."""
    statements = _with_output(
        {"cmd": "ANGLE", "name": "a", "min_deg": 320.0, "max_deg": 40.0},
        {
            "cmd": "SPHERICALANGLE",
            "name": "sph",
            "min_theta_deg": -80.0,
            "max_theta_deg": 80.0,
            "min_phi_deg": 320.0,
            "max_phi_deg": 40.0,
        },
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=4)
    for param in ("a@theta", "sph@phi"):
        assert sampler._space[param] == D.FloatDistribution(320.0, 400.0)

    def objective(p):
        angles = (p["a"], p["sph"]["phi"])
        return {"out": sum(abs(math.sin(math.radians(v))) for v in angles)}

    study, caught, fallbacks = _run_counting_fallbacks(
        sampler,
        objective,
        20,
        enqueue={"a@theta": 370.0, "sph@theta": 0.0, "sph@phi": 390.0},
    )
    assert caught == [] and fallbacks == []
    first = study.trials[0]
    assert first.state.name == "COMPLETE"
    assert first.params["a@theta"] == 370.0 and first.params["sph@phi"] == 390.0
    for t in study.trials:
        for param in ("a@theta", "sph@phi"):
            assert 320.0 <= t.params[param] <= 400.0
    fold = sampler._var_codecs["a"].from_engine["a@theta"]
    assert fold(10.0) == 370.0 and fold(320.0) == 320.0 and fold(40.0) == 400.0


def test_discrete_statements_use_optunas_grid_top_for_the_engine_too():
    """From 0 to 1 with step 0.6, the engine's own grid is 0, 0.6, 1.2 and Optuna's is 0, 0.6.
    With statements=, the sampler gives the engine Optuna's top, so no engine value is
    rejected and the engine's grid is Optuna's grid."""
    statements = _with_output(
        {"cmd": "DISCRETE", "name": "d", "min": 0.0, "max": 1.0, "step": 0.6},
        {"cmd": "DISCRETEVECTOR", "name": "v", "size": 2, "min": 0.0, "max": 1.0, "step": 0.6},
        direction="max",
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=4)
    assert sampler._space["d"] == D.FloatDistribution(0.0, 0.6, step=0.6)
    assert sampler._space["v@1"] == D.FloatDistribution(0.0, 0.6, step=0.6)
    study, caught, fallbacks = _run_counting_fallbacks(
        sampler, lambda p: {"out": p["d"] + sum(p["v"])}, 20
    )
    assert caught == [] and fallbacks == []
    assert {t.params["d"] for t in study.trials if t.params} == {0.0, 0.6}
    assert statements[0]["max"] == 1.0  # the caller's statement is not changed


def test_discrete_model_mode_registers_the_engines_grid():
    """With model=, the caller built the engine model, so the registered distribution is the
    engine's grid, 0, 0.6, 1.2, and the engine's 1.2 is not rejected."""
    model = optquest.OptQuestModel()
    model.set_license("")
    model.add_discrete_variable("d", 0.0, 1.0, 0.6)
    model.add_output_variable("out")
    model.add_maximize_objective("obj", "out")
    sampler = OptQuestSampler(model=model, seed=4)
    assert sampler._space["d"] == D.FloatDistribution(0.0, 1.2, step=0.6)
    study, caught, fallbacks = _run_counting_fallbacks(sampler, lambda p: {"out": p["d"]}, 10)
    assert fallbacks == []
    assert {t.params["d"] for t in study.trials if t.params} == {0.0, 0.6, 1.2}


# ---------------------------------------------------------------------------------------------
# Engine values that Optuna must keep (review of #437, round 2)
# ---------------------------------------------------------------------------------------------


def test_discrete_model_mode_keeps_the_engines_binary_float_top():
    """From 0 to 1 with step 0.4, the engine's top is 0 + 3 * 0.4 = 1.2000000000000002 in binary
    floating point; the registered top is 1.2. The value is moved onto the bound."""
    model = optquest.OptQuestModel()
    model.set_license("")
    model.add_discrete_variable("d", 0.0, 1.0, 0.4)
    model.add_output_variable("out")
    model.add_maximize_objective("obj", "out")
    sampler = OptQuestSampler(model=model, seed=4)
    assert sampler._space["d"] == D.FloatDistribution(0.0, 1.2, step=0.4)
    study, caught, fallbacks = _run_counting_fallbacks(sampler, lambda p: {"out": p["d"]}, 12)
    assert caught == [] and fallbacks == []
    assert {t.params["d"] for t in study.trials if t.params} == {0.0, 0.4, 0.8, 1.2}


def test_a_one_step_discrete_grid_is_accepted_in_statements_mode():
    statements = _with_output(
        {"cmd": "DISCRETE", "name": "d", "min": -1.0, "max": -0.85, "step": 0.1},
        direction="max",
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=4)
    assert sampler._space["d"] == D.FloatDistribution(-1.0, -0.9, step=0.1)
    study, caught, fallbacks = _run_counting_fallbacks(sampler, lambda p: {"out": p["d"]}, 4)
    assert fallbacks == []
    assert {t.params["d"] for t in study.trials if t.params} == {-1.0, -0.9}


def _optuna_top(low, high, step):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Optuna warns that it moves `high` onto its grid
        return D.FloatDistribution(low, high, step=step).high


@pytest.mark.parametrize(
    "low, high, step",
    [
        (0.0, 1.0, 1.0 / 13.0),
        (0.0, 1.0, 1.0 / 15.0),
        (0.0, 3.0, 1.0 / 3.0),
        (0.0, 10.0, 1.0 / 3.0),
        (0.0, 1.0, 0.1 * 3),
        (-1.53, -0.93, 1.0 / 3.0),
        (1.0, 1.2285714285714286, 1.0 / 7.0),
    ],
)
def test_a_long_step_reaches_optunas_grid_top(low, high, step):
    """With statements=, the largest value is the top of Optuna's own FloatDistribution(min,
    max, step=step), and the engine proposes it: a search for the largest value finds it, and
    Optuna keeps every value that the engine proposes."""
    statements = _with_output(
        {"cmd": "DISCRETE", "name": "d", "min": low, "max": high, "step": step}, direction="max"
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=4)
    study, caught, fallbacks = _run_counting_fallbacks(sampler, lambda p: {"out": p["d"]}, 40)
    assert caught == [] and fallbacks == []
    values = {t.params["d"] for t in study.trials if t.params}
    top = _optuna_top(low, high, step)
    assert max(values) == top
    for v in values:  # every value is on the grid
        k = (v - low) / step
        assert low <= v <= top and abs(k - round(k)) < 1e-8, v


@pytest.mark.parametrize(
    "high, step",
    [(1.0, 0.333333), (1.0, 0.142858), (100.000001, 1.0), (4.999999, 1.0), (1.0, 0.3333333)],
)
def test_a_built_model_keeps_the_max_the_engine_keeps(high, step):
    """With model=, the engine keeps `max` as its largest value when the grid value nearest to
    it is within 1e-5 of it (from 0 to 1 with step 0.333333, it proposes 1.0, and 0.999999 too),
    so the sampler registers the range up to `max`, and Optuna keeps every value."""
    model = optquest.OptQuestModel()
    model.set_license("")
    model.add_discrete_variable("d", 0.0, high, step)
    model.add_output_variable("out")
    model.add_maximize_objective("obj", "out")
    sampler = OptQuestSampler(model=model, seed=4)
    assert sampler._space["d"].high == high
    study, caught, fallbacks = _run_counting_fallbacks(sampler, lambda p: {"out": p["d"]}, 30)
    assert caught == [] and fallbacks == []
    assert max(t.params["d"] for t in study.trials if t.params) == high


@pytest.mark.parametrize("low, high, step", [(1e6, 1e6 + 10.0, 0.001), (1e7, 1e7 + 1000.0, 0.01)])
def test_a_fine_step_far_from_zero_keeps_the_engine_values(low, high, step):
    """Far from zero for its step, the engine's grid values are too far from Optuna's grid in
    binary floating point for Optuna's tolerance, so the sampler registers the range."""
    statements = _with_output(
        {"cmd": "DISCRETE", "name": "d", "min": low, "max": high, "step": step}
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=4)
    assert sampler._space["d"].step is None
    study, caught, fallbacks = _run_counting_fallbacks(
        sampler, lambda p: {"out": abs(p["d"] - (low + 3.37))}, 25
    )
    assert caught == [] and fallbacks == []


@pytest.mark.parametrize(
    "low, high, step", [(0.1, 0.3, 0.2), (-1.0, -0.9, 0.1), (2.5, 2.8, 0.3), (1.0, 1.2, 0.2)]
)
def test_a_two_value_grid_is_accepted_in_every_mode(low, high, step):
    """0.3 - 0.1 is 0.19999999999999998 in binary floating point, shorter than the step 0.2,
    and the engine refuses a range shorter than one step; the engine gets a `max` a few ulps
    higher, and its two values are Optuna's two values."""
    top = _optuna_top(low, high, step)
    plain = OptQuestSampler(
        search_space={"d": D.FloatDistribution(low, high, step=step)},
        directions=[MAX],
        license="",
        seed=4,
    )
    statements = OptQuestSampler(
        statements=_with_output(
            {"cmd": "DISCRETE", "name": "d", "min": low, "max": high, "step": step},
            direction="max",
        ),
        license="",
        seed=4,
    )
    for sampler, objective in (
        (plain, lambda p: (p["d"],)),
        (statements, lambda p: {"out": p["d"]}),
    ):
        study, caught, fallbacks = _run_counting_fallbacks(sampler, objective, 6)
        assert caught == [] and fallbacks == []
        assert {t.params["d"] for t in study.trials if t.params} == {low, top}


def test_suggest_all_keeps_a_long_step_grid_in_plain_mode():
    """In plain mode, suggest_all() and wrap() call trial.suggest_float() with the registered
    distribution. For a step such as 1/3, Optuna moves the `high` that it already moved onto
    its grid down one more step, so the sampler registers the range of the grid instead, as in
    the other modes. A hand-written call with the caller's bounds keeps the stepped grid."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Optuna warns that it moves `high` onto its grid
        space = {"d": D.FloatDistribution(0.0, 3.0, step=1.0 / 3.0)}
    sampler = OptQuestSampler(search_space=space, directions=[MAX], license="", seed=4)
    assert sampler._space["d"] == D.FloatDistribution(0.0, space["d"].high)
    assert space["d"].step is not None  # the caller's dict is not changed
    study, caught, fallbacks = _run_counting_fallbacks(sampler, lambda p: (p["d"],), 30)
    assert caught == [] and fallbacks == []
    assert max(t.params["d"] for t in study.trials if t.params) == space["d"].high

    sampler = OptQuestSampler(search_space=space, directions=[MAX], license="", seed=4)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    fallbacks = []
    original = sampler.sample_independent
    sampler.sample_independent = lambda *a: fallbacks.append(a[2]) or original(*a)
    caught = _optimize(study, lambda t: t.suggest_float("d", 0.0, 3.0, step=1.0 / 3.0), 30)
    assert [w for w in caught if "discarded" in w] == []  # Optuna warns about its `high`
    assert fallbacks == []
    assert max(t.params["d"] for t in study.trials if t.params) == space["d"].high


def test_region_points_on_the_boundary_are_kept():
    """The engine's point on the edge of a region can carry floating-point noise below a bound
    (0.09999999999999998 for 0.1); it is moved onto the bound."""
    statements = _with_output(
        {"cmd": "REGION", "name": "r", "vertices": [[0.1, 0.1], [0.7, 0.2], [0.3, 0.9]]},
        {
            "cmd": "SPHERICALREGION",
            "name": "g",
            "vertices": [
                [-21.682, 93.379],
                [-21.682, 111.805],
                [-36.316, 111.805],
                [-36.316, 93.379],
            ],
        },
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=1)
    study, caught, fallbacks = _run_counting_fallbacks(
        sampler, lambda p: {"out": p["r"]["x"] + p["r"]["y"] + p["g"]["phi"]}, 40
    )
    assert caught == [] and fallbacks == []


def test_a_discarded_engine_value_is_reported_once_for_each_parameter():
    sampler = _sphere_sampler()
    sampler._from_engine["x"] = lambda raw: 99.0  # outside FloatDistribution(-5, 10)
    study = optuna.create_study(sampler=sampler)
    caught = _optimize(study, sphere, 3)
    assert len([w for w in caught if "discarded the engine's value 99.0" in w]) == 1


@pytest.mark.parametrize(
    "stmt, error",
    [
        ({"cmd": "INTEGER", "name": "v", "min": 0.5, "max": 5.5}, ValueError),
        ({"cmd": "CONTINUOUS", "name": "x", "min": 0.0, "mxa": 1.0}, optquest.EmitError),
        ({"cmd": "CONTINUOUS", "name": "e", "min": 0.0, "max": 1.0}, optquest.EmitError),
        ({"cmd": "NORMALRANDOM", "name": "r", "mean": 0.0, "sigma": 1.0}, NotImplementedError),
    ],
)
def test_statements_that_would_build_a_different_model_are_refused(stmt, error):
    with pytest.raises(error):
        OptQuestSampler(statements=_with_output(stmt), license="", seed=1)


X_STMT = {"cmd": "CONTINUOUS", "name": "x", "min": -1.0, "max": 1.0}


def _x_points(**kwargs):
    statements = kwargs.pop("statements")
    sampler = OptQuestSampler(statements=statements, **kwargs)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    _optimize(study, sampler.wrap(lambda p: {"out": p["x"] ** 2}), 16)
    return sampler, [t.params["x"] for t in study.trials]


def test_a_randomseed_statement_applies_unless_another_seed_differs():
    seeded = _with_output(X_STMT, {"cmd": "RANDOMSEED", "seed": 7})
    sampler, points = _x_points(statements=seeded, license="")
    assert "RANDOMSEED 7" in sampler._model.describe().splitlines()
    assert points == _x_points(statements=_with_output(X_STMT), license="", seed=7)[1]
    assert points != _x_points(statements=_with_output(X_STMT), license="")[1]
    assert _x_points(statements=seeded, license="", seed=7)[1] == points
    with pytest.raises(ValueError, match=r"RANDOMSEED statement \(7\) and the seed= argument"):
        OptQuestSampler(statements=seeded, license="", seed=8)
    twice = _with_output(
        X_STMT, {"cmd": "RANDOMSEED", "seed": 3}, {"cmd": "RANDOMSEED", "seed": 4}
    )
    with pytest.raises(ValueError, match="and another RANDOMSEED statement"):
        OptQuestSampler(statements=twice, license="")


def test_a_license_statement_applies_unless_another_key_differs(monkeypatch):
    """The key of a LICENSE statement reaches set_license(). The test refuses it there, which
    stops the sampler before the engine's 10-second check of a key that is not a license."""
    keys = []

    def refuse(model, key):
        keys.append(key)
        return False

    monkeypatch.setattr(optquest.OptQuestModel, "set_license", refuse)
    with pytest.raises(RuntimeError, match="refused the license key"):
        OptQuestSampler(statements=_with_output(X_STMT, {"cmd": "LICENSE", "license_id": "K-1"}))
    assert keys == ["K-1"]
    monkeypatch.undo()
    with pytest.raises(ValueError, match="LICENSE statement and the license= argument"):
        OptQuestSampler(
            statements=_with_output(X_STMT, {"cmd": "LICENSE", "license_id": "A"}), license="B"
        )
    twice = _with_output(
        X_STMT, {"cmd": "LICENSE", "license_id": "A"}, {"cmd": "LICENSE", "license_id": "B"}
    )
    with pytest.raises(ValueError, match="and another LICENSE statement"):
        OptQuestSampler(statements=twice)


def _xmodel():
    model = optquest.OptQuestModel()
    model.set_license("")
    model.add_continuous_variable("x", 0.0, 1.0)
    model.add_output_variable("out")
    model.add_minimize_objective("obj", "out")
    return model


def test_a_license_key_the_engine_refuses_raises_in_every_mode():
    """set_license() answers False, and does not raise, for a key with whitespace (a key read
    from a file keeps its line break) and on a model that is already initialized; the engine
    would then run in demo mode with no error."""
    key = "ABC-123\n"
    with pytest.raises(RuntimeError, match="strip"):
        OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN], license=key)
    with pytest.raises(RuntimeError, match="strip"):
        OptQuestSampler(statements=_with_output(X_STMT), license=key)
    with pytest.raises(RuntimeError, match="strip"):
        OptQuestSampler(model=_xmodel(), license=key)
    model = _xmodel()
    assert model.initialize()
    with pytest.raises(RuntimeError, match="already initialized"):
        OptQuestSampler(model=model, license="KEY")


def test_model_mode_refuses_what_it_cannot_take_before_initializing_the_model():
    """The sampler refuses an Adaptive Sampling command, NOGENERATE, bounds or a step that the
    codec cannot register, and an EXCLUSIVE edge that the engine would propose but the variable
    cannot take, before it initializes the caller's model: the model can still be changed."""
    model = _xmodel()
    model.set_sample_method(1)
    with pytest.raises(NotImplementedError, match="SAMPLE"):
        OptQuestSampler(model=model)
    model = _xmodel()
    model.suggested_solutions_only()
    with pytest.raises(NotImplementedError, match="NOGENERATE"):
        OptQuestSampler(model=model)
    assert model.set_random_seed(3)  # not initialized
    model = _xmodel()
    model.add_integer_variable("v", 0.5, 5.5)
    with pytest.raises(ValueError, match="whole numbers"):
        OptQuestSampler(model=model)
    model = _xmodel()
    model.add_discrete_variable("d", 0.0, 1.0, -0.25)
    with pytest.raises(ValueError, match="step must be a positive number"):
        OptQuestSampler(model=model)
    model = _xmodel()
    model.add_integer_variable("v", 0, 20)
    model.add_variable_exclusive_range("v", 5.5, 15.5)
    with pytest.raises(ValueError, match="from 5.0 to 16.0"):
        OptQuestSampler(model=model)
    assert model.set_random_seed(3)  # not initialized


def test_model_mode_keeps_random_variables_and_the_models_own_seed():
    model = _xmodel()
    model.add_normal_random_variable("r", 0.0, 1.0)
    sampler = OptQuestSampler(model=model, seed=1)
    assert set(sampler._space) == {"x"}  # the engine draws the random variable

    def free_value(model_seed):
        """The value that the fallback sampler draws for the free parameter of a partially
        enqueued point: a RANDOMSEED on the model seeds it as seed= would."""
        model = optquest.OptQuestModel()
        model.set_license("")
        model.add_continuous_variable("x", 0.0, 1.0)
        model.add_continuous_variable("y", 0.0, 1.0)
        model.add_output_variable("out")
        model.add_minimize_objective("obj", "out")
        model.set_random_seed(model_seed)
        sampler = OptQuestSampler(model=model)
        study = optuna.create_study(sampler=sampler, directions=sampler.directions)
        study.enqueue_trial({"x": 0.5})
        _optimize(study, sampler.wrap(lambda p: {"out": p["x"] + p["y"]}), 1)
        return study.trials[0].params["y"]

    assert free_value(5) == free_value(5) != free_value(6)


def test_ask_and_tell_ends_with_a_pruned_trial_at_the_demo_cap():
    """study.stop() works only inside study.optimize(); with ask-and-tell, the suggest_*() call
    that reaches the end raises optuna.TrialPruned instead of a RuntimeError."""
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(55):
            trial = study.ask()
            try:
                value = sphere(trial)
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                break
            study.tell(trial, value)
    assert _states(study) == {"COMPLETE": 50, "PRUNED": 1}


def test_plain_mode_takes_names_the_engine_cannot_and_a_tuple_of_directions():
    space = {
        "e": D.FloatDistribution(-1.0, 1.0),
        "learning rate": D.FloatDistribution(1e-4, 1e-1, log=True),
        "x": D.FloatDistribution(-1.0, 1.0),
        "X": D.IntDistribution(0, 3),
        "1st": D.CategoricalDistribution(["a", "b"]),
    }
    sampler = OptQuestSampler(search_space=space, directions=("minimize",), license="", seed=2)

    def objective(trial):
        values = {name: _suggest(trial, name, dist) for name, dist in space.items()}
        return values["e"] ** 2 + values["x"] ** 2 + values["X"]

    study = optuna.create_study(sampler=sampler)
    fallbacks = []
    original = sampler.sample_independent

    def counting(study, trial, name, dist):
        fallbacks.append(name)
        return original(study, trial, name, dist)

    sampler.sample_independent = counting
    assert _optimize(study, objective, 8) == []
    assert fallbacks == [] and _states(study) == {"COMPLETE": 8}
    assert set(study.trials[0].params) == set(space)


def test_plain_mode_takes_the_function_names_that_the_binding_does_not_list():
    """The 10.2.0.0 engine reserves function names that optquest.RESERVED_NAMES does not have
    (step1, normcdf, lookup1, ...): a parameter with such a name reached the engine under that
    name, and the engine refused the model."""
    unlisted = importlib.import_module(f"{_package}.OptQuestSampler")._UNLISTED_RESERVED_NAMES
    assert {"step1", "normcdf", "lookup1", "tcdf"} <= unlisted
    space = {name: D.FloatDistribution(0.0, 1.0) for name in sorted(unlisted)}
    space["Step1"] = D.FloatDistribution(0.0, 1.0)  # the engine compares without regard to case
    sampler = OptQuestSampler(search_space=space, directions=[MIN], license="", seed=2)
    assert set(sampler._engine_names) == set(space)
    study = optuna.create_study(sampler=sampler)
    fallbacks = []
    original = sampler.sample_independent

    def counting(study, trial, name, dist):
        fallbacks.append(name)
        return original(study, trial, name, dist)

    sampler.sample_independent = counting
    caught = _optimize(study, lambda t: sum(t.suggest_float(n, 0.0, 1.0) for n in space), 3)
    assert caught == [] and fallbacks == []
    assert set(study.trials[0].params) == set(space)


def _suggest(trial, name, dist):
    if isinstance(dist, D.CategoricalDistribution):
        return trial.suggest_categorical(name, dist.choices)
    if isinstance(dist, D.IntDistribution):
        return trial.suggest_int(name, dist.low, dist.high)
    return trial.suggest_float(name, dist.low, dist.high, log=dist.log)


def test_constructor_refuses_inputs_of_the_wrong_type():
    with pytest.raises(TypeError, match="search_space"):
        OptQuestSampler(search_space=[("x", SPHERE_SPACE["x"])], directions=[MIN])
    with pytest.raises(TypeError, match="OptQuestModel"):
        OptQuestSampler(model="not a model")


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_a_wrapper_that_fixes_a_parameter_gets_a_copy_of_the_search_space():
    sampler = OptQuestSampler(statements=CONSTRAINED, license="", seed=5)
    fixed = optuna.samplers.PartialFixedSampler({"x": 1.0}, sampler)
    study = optuna.create_study(sampler=fixed, directions=sampler.directions)

    def sim(p):
        return {"score": p["x"] ** 2 + p["y"] ** 2, "budget": p["x"] + p["y"]}

    assert _optimize(study, sampler.wrap(sim), 5) == []
    assert _states(study) == {"COMPLETE": 5}
    assert {t.params["x"] for t in study.trials} == {1.0}
    assert "x" in sampler._space


def test_a_direction_mismatch_raises_every_time():
    sampler = OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN], license="", seed=3)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    for _ in range(2):
        with pytest.raises(ValueError, match="direction"):
            study.optimize(sphere, n_trials=1)


def test_an_objective_may_name_its_output_in_another_case():
    statements = [
        {"cmd": "CONTINUOUS", "name": "x", "min": -1.0, "max": 1.0},
        {"cmd": "OUTPUT", "name": "score"},
        {"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "SCORE"},
    ]
    sampler = OptQuestSampler(statements=statements, license="", seed=1)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"score": p["x"] ** 2}), 3) == []
    assert _states(study) == {"COMPLETE": 3}


def _side_values(point, vertices):
    """dot(P, A x B) for each edge A -> B. The region of a clockwise polygon is where every
    value is negative or zero."""

    def unit(lat, lon):
        la, lo = math.radians(lat), math.radians(lon)
        return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))

    p = unit(*point)
    u = [unit(*v) for v in vertices]
    values = []
    for a, b in zip(u, u[1:] + u[:1]):
        n = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        values.append(sum(x * y for x, y in zip(p, n)))
    return values


@pytest.mark.parametrize(
    "vertices, inside",
    [
        ([[20, 10], [20, 20], [10, 20], [10, 10]], "box"),
        ([[10, 10], [10, 20], [20, 20], [20, 10]], "rest"),
        ([[60, 0], [60, -90], [60, 180], [60, 90]], "cap"),
    ],
)
def test_spherical_polygon_points_from_the_engine_fit_the_distribution(vertices, inside):
    statements = [
        {"cmd": "SPHERICALREGION", "name": "p", "vertices": vertices},
        {"cmd": "OUTPUT", "name": "out"},
        {"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "out"},
    ]
    sampler = OptQuestSampler(statements=statements, license="", seed=4)
    fallbacks = []
    original = sampler.sample_independent

    def counting(study, trial, name, dist):
        fallbacks.append(name)
        return original(study, trial, name, dist)

    sampler.sample_independent = counting
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": p["p"]["phi"]}), 40) == []
    assert fallbacks == []
    box = [[20, 10], [20, 20], [10, 20], [10, 10]]
    for t in study.trials:
        point = (t.params["p@theta"], t.params["p@phi"])
        if inside == "box":
            assert max(_side_values(point, box)) <= 1e-8, point
        elif inside == "rest":
            assert max(_side_values(point, box)) >= -1e-8, point
        else:
            assert point[0] >= 60.0 - 1e-6, point


# ---------------------------------------------------------------------------------------------
# Review of #437, round 3
# ---------------------------------------------------------------------------------------------


def test_int_parameters_are_ints_in_trial_params():
    """The engine reports every value as a float. Optuna keeps the value that the sampler gives
    it in trial.params, so an objective that reads trial.params after suggest_all() would get
    3.0; the sampler gives an int, as Optuna's own samplers do."""
    statements = _with_output(
        {"cmd": "INTEGER", "name": "k", "min": 0, "max": 3},
        {"cmd": "BINARY", "name": "b"},
        {"cmd": "PERMUTATION", "name": "perm", "items": ["p", "q", "r"]},
    )
    costs = [4.0, 1.0, 3.0, 2.0]
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)

    def objective(trial):
        sampler.suggest_all(trial)
        return costs[trial.params["k"]] + trial.params["b"] + trial.params["perm@p"]

    assert _optimize(study, objective, 8) == []
    for t in study.trials:
        assert all(type(t.params[name]) is int for name in ("k", "b", "perm@p", "perm@q"))
    space = {"i": D.IntDistribution(1, 10, step=3), "j": D.IntDistribution(0, 5)}
    sampler = OptQuestSampler(search_space=space, directions=[MIN], license="", seed=3)
    study = optuna.create_study(sampler=sampler)
    assert (
        _optimize(study, lambda t: t.suggest_int("i", 1, 10, step=3) + t.suggest_int("j", 0, 5), 6)
        == []
    )
    assert all(type(v) is int for t in study.trials for v in t.params.values())


def test_statements_take_numpy_values_and_numbers_given_as_text():
    """optquest.emit_model() refuses a numpy array for a list field, and the typed model methods
    refuse 30.0 or "30" for an int field that emit_model() accepts; the sampler converts both."""
    np = pytest.importorskip("numpy")
    statements = _with_output(
        {"cmd": "ENUMERATION", "name": "en", "values": np.linspace(0.0, 1.0, 5)},
        {"cmd": "DISCRETE", "name": "d", "min": np.float64(0), "max": np.float64(1), "step": 0.25},
        {"cmd": "INTEGER", "name": "i", "min": "0", "max": np.int64(4)},
        {"cmd": "SEQUENCE", "name": "s", "components": ["a", "b", "c"], "top_k": 2.0},
        {"cmd": "STOPTIME", "seconds": 30.0},
        {"cmd": "STOPCYCLES", "cycles": "50"},
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    assert sampler._space["en"] == D.CategoricalDistribution([0.0, 0.25, 0.5, 0.75, 1.0])
    assert sampler._space["i"] == D.IntDistribution(0, 4)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    caught = _optimize(study, sampler.wrap(lambda p: {"out": p["en"] + p["d"] + p["i"]}), 5)
    assert caught == [] and _states(study) == {"COMPLETE": 5}
    assert isinstance(statements[0]["values"], np.ndarray)  # the caller's statement is not changed


def test_a_sampler_searches_for_one_study():
    """The engine keeps the state of one search, and trial numbers are unique only within one
    study, so the results of a second study would never reach the engine."""
    sampler = _sphere_sampler()
    first = optuna.create_study(sampler=sampler)
    assert _optimize(first, sphere, 3) == []
    second = optuna.create_study(sampler=sampler)
    with pytest.raises(RuntimeError, match="Create a new OptQuestSampler"):
        second.optimize(sphere, n_trials=1)
    assert _optimize(first, sphere, 2) == []
    assert sampler._model.get_number_of_completed_iterations() == 5


def test_an_external_point_without_every_parameter_is_not_sent():
    """The engine takes a variable with no value as 0.0, even outside its bounds, so a point
    added without one of the sampler's parameters is not sent, with one warning."""
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    for x in (1.0, 2.0):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": x}, distributions={"x": SPHERE_SPACE["x"]}, value=-100.0
            )
        )
    caught = _optimize(study, sphere, 3)
    assert len(caught) == 1 and "no value for the parameter(s) ['y']" in caught[0]
    assert len(sampler._model.get_all_solutions()) == 3  # the two added points were not sent


def test_an_external_value_that_the_model_cannot_take_is_not_sent():
    """A value from another run that this model cannot take, such as the log of a negative
    number, is reported with a warning instead of raising from sample_relative()."""
    space = {"f": D.FloatDistribution(1e-3, 10.0, log=True)}
    sampler = OptQuestSampler(search_space=space, directions=[MIN], license="", seed=3)
    study = optuna.create_study(sampler=sampler)
    study.add_trial(
        optuna.trial.create_trial(
            params={"f": -0.5}, distributions={"f": D.FloatDistribution(-1.0, 1.0)}, value=1.0
        )
    )
    caught = _optimize(study, lambda t: t.suggest_float("f", 1e-3, 10.0, log=True), 3)
    assert len(caught) == 1 and "cannot accept (f=-0.5)" in caught[0]
    assert "did not send it to the engine" in caught[0]
    assert len(sampler._model.get_all_solutions()) == 3


def test_a_parameter_with_the_name_of_an_internal_output_gets_an_internal_name():
    space = {"OQout00": D.FloatDistribution(0.0, 1.0), "oqobj00": D.FloatDistribution(0.0, 1.0)}
    sampler = OptQuestSampler(search_space=space, directions=[MIN], license="", seed=3)
    described = sampler._model.describe().splitlines()
    assert "OUTPUT OQout00" in described and "OBJECTIVE OQobj00 MIN OQout00" in described
    assert set(sampler._engine_names) == {"OQout00", "oqobj00"}
    study = optuna.create_study(sampler=sampler)

    def objective(trial):
        return trial.suggest_float("OQout00", 0.0, 1.0) + 2 * trial.suggest_float(
            "oqobj00", 0.0, 1.0
        )

    assert _optimize(study, objective, 5) == []
    for sol, trial in zip(
        sorted(sampler._model.get_all_solutions(), key=lambda s: s.get_iteration()), study.trials
    ):
        assert math.isclose(sol.get_value("OQout00"), trial.value)


def test_an_exclusive_range_with_edges_the_variable_cannot_take_keeps_its_values():
    """The engine moves a value inside an excluded range to the nearer edge. With statements=,
    an edge that the variable cannot take moves outward to the nearest value that it can take,
    which excludes the same values, so Optuna keeps every value and none is in the range."""
    statements = _with_output(
        {"cmd": "INTEGER", "name": "v", "min": 0, "max": 20},
        {"cmd": "EXCLUSIVE", "name": "v", "min": 5.5, "max": 15.5},
        {"cmd": "ENUMERATION", "name": "k", "values": [1.0, 2.0, 3.0]},
        {"cmd": "EXCLUSIVE", "name": "k", "min": 1.5, "max": 2.5},
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    described = sampler._model.describe().splitlines()
    assert "EXCLUSIVE v 5 16" in described and "EXCLUSIVE k 1 3" in described
    study, caught, fallbacks = _run_counting_fallbacks(
        sampler, lambda p: {"out": (p["v"] - 10) ** 2 + (p["k"] - 2.0) ** 2}, 30
    )
    assert fallbacks == [] and [w for w in caught if "discarded" in w] == []
    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    # The engine enumerates the 11 values of v outside 6..15 times the 2 values of k, once each.
    points = {(t.params["v"], t.params["k"]) for t in complete}
    assert len(points) == len(complete) == 22
    assert not any(6 <= v <= 15 for v, _ in points)
    assert {k for _, k in points} == {1.0, 3.0}


def test_a_mixture_with_a_component_count_is_built_as_the_engine_builds_it():
    """A count n in place of the component names gives the components 1 to n, as the engine
    names them when it reads a MIXTURE line with a count; a count below 1 is refused."""
    statements = _with_output({"cmd": "MIXTURE", "name": "m", "components": 3, "sum_to": 2.0})
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    assert set(sampler._space) == {"m@1", "m@2", "m@3"}
    assert any(
        line.startswith("MIXTURE m 1,2,3") for line in sampler._model.describe().splitlines()
    )
    seen = []

    def sim(p):
        seen.append(p["m"])
        return {"out": p["m"]["1"]}

    study, caught, fallbacks = _run_counting_fallbacks(sampler, sim, 8)
    assert caught == [] and fallbacks == [] and _states(study) == {"COMPLETE": 8}
    for mix in seen:
        assert set(mix) == {"1", "2", "3"} and math.isclose(sum(mix.values()), 2.0, abs_tol=1e-9)
    for count in (0, -2):
        statements = _with_output({"cmd": "MIXTURE", "name": "m", "components": count})
        with pytest.raises(ValueError, match="count of at least 1"):
            OptQuestSampler(statements=statements, license="")


def test_model_mode_refuses_a_seed_or_a_key_other_than_the_models_own():
    """As for a RANDOMSEED or LICENSE statement in statements=, the seed= and license=
    arguments must not give the engine another seed or key than the one set on the model."""
    model = _xmodel()
    model.set_random_seed(5)
    with pytest.raises(
        ValueError, match=r"the seed set on the model \(5\) and the seed= argument \(9\)"
    ):
        OptQuestSampler(model=model, seed=9)
    sampler = OptQuestSampler(model=model, seed=5)  # the same seed
    assert "RANDOMSEED 5" in sampler._model.describe().splitlines()
    model = _xmodel()
    model.set_license("K-1")
    with pytest.raises(ValueError, match="license key set on the model and the license= argument"):
        OptQuestSampler(model=model, license="K-2")
    assert model.set_random_seed(3)  # the model was not initialized


# ---------------------------------------------------------------------------------------------
# One study, refusals before the model is changed, and the seed and key of a model
# ---------------------------------------------------------------------------------------------


def test_a_second_study_with_the_same_name_is_refused():
    """A study created again with the same name (a notebook cell run twice) has the numbers of
    trials that the sampler already finished, so it is refused as a second study is. A study
    loaded again from its storage goes on."""
    sampler = _sphere_sampler()
    first = optuna.create_study(study_name="demo", sampler=sampler)
    assert _optimize(first, sphere, 3) == []
    second = optuna.create_study(study_name="demo", sampler=sampler)
    finished = []

    def objective(trial):
        finished.append(sphere(trial))
        return finished[-1]

    with pytest.raises(RuntimeError, match="Create a new OptQuestSampler"):
        second.optimize(objective, n_trials=1)
    assert finished == []  # refused when the trial asks for its parameters
    storage = optuna.storages.InMemoryStorage()
    sampler = _sphere_sampler()
    kept = optuna.create_study(study_name="kept", storage=storage, sampler=sampler)
    assert _optimize(kept, sphere, 3) == []
    again = optuna.load_study(study_name="kept", storage=storage, sampler=sampler)
    assert _optimize(again, sphere, 2) == []
    assert sampler._model.get_number_of_completed_iterations() == 5


def test_the_engine_handshake_is_serialized_between_threads(monkeypatch):
    """While one thread sends a finished trial to the engine (after_trial holds the lock until
    submit() returns), a trial in another thread that samples its parameters waits for the lock
    and does not reach create_solution()."""
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    first = study.ask()
    value = sphere(first)
    # Optuna samples the relative parameters of a trial at its first suggest_*() call, not in
    # ask(), so the second trial samples them only in the asker's thread.
    second = study.ask()
    in_submit, release = threading.Event(), threading.Event()
    created = []
    submit = optquest.OptQuestSolution.submit
    create_solution = optquest.OptQuestModel.create_solution

    def blocking_submit(solution):
        in_submit.set()
        release.wait(10)
        return submit(solution)

    def counting_create_solution(model):
        created.append(threading.current_thread().name)
        return create_solution(model)

    monkeypatch.setattr(optquest.OptQuestSolution, "submit", blocking_submit)
    monkeypatch.setattr(optquest.OptQuestModel, "create_solution", counting_create_solution)
    teller = threading.Thread(target=lambda: study.tell(first, value), name="teller")
    asker = threading.Thread(target=lambda: sphere(second), name="asker")
    teller.start()
    try:
        assert in_submit.wait(10)
        asker.start()
        asker.join(0.5)
        assert created == []  # the asker waits for the lock that the teller holds
    finally:
        release.set()
        teller.join(10)
        if asker.ident is not None:
            asker.join(10)
    assert not teller.is_alive() and not asker.is_alive()
    assert created == ["asker"]


def test_two_trials_do_not_sample_from_the_engine_at_the_same_time(monkeypatch):
    """sample_relative() holds the lock for its whole handshake with the engine: a second trial
    that enters it while the first is in create_solution() waits for the first to finish."""
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    trial_a, trial_b = study.ask(), study.ask()
    b_entered, b_go, a_inside, a_release = (threading.Event() for _ in range(4))
    created = []
    create_solution = optquest.OptQuestModel.create_solution
    sample_relative = sampler.sample_relative

    def gated_sample_relative(study, trial, search_space):
        # Hold B at the entry of sample_relative(), after Optuna's other sampler calls.
        if threading.current_thread().name == "B":
            b_entered.set()
            b_go.wait(10)
        return sample_relative(study, trial, search_space)

    def blocking_create_solution(model):
        created.append(threading.current_thread().name)
        if threading.current_thread().name == "A":
            a_inside.set()
            a_release.wait(10)
        return create_solution(model)

    monkeypatch.setattr(sampler, "sample_relative", gated_sample_relative)
    monkeypatch.setattr(optquest.OptQuestModel, "create_solution", blocking_create_solution)
    thread_b = threading.Thread(target=lambda: sphere(trial_b), name="B")
    thread_a = threading.Thread(target=lambda: sphere(trial_a), name="A")
    thread_b.start()
    try:
        assert b_entered.wait(10)
        thread_a.start()
        assert a_inside.wait(10)
        b_go.set()
        thread_b.join(0.5)
        assert created == ["A"]  # B waits for the lock that A holds in sample_relative()
    finally:
        b_go.set()
        a_release.set()
        thread_b.join(10)
        if thread_a.ident is not None:
            thread_a.join(10)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert created == ["A", "B"]


def test_a_finite_space_that_runs_out_warns_that_it_is_exhausted():
    """When the engine has proposed every point of a finite space (CannotGenerate), the sampler
    prunes the trial, stops the study, and says that the space is exhausted."""
    sampler = OptQuestSampler(
        statements=_with_output({"cmd": "CHOICE", "name": "c", "k": 3}), license="", seed=3
    )
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    caught = _optimize(study, sampler.wrap(lambda p: {"out": float(p["c"])}), 20)
    assert sampler._model.get_termination_reason() == optquest.OQTerminationReasonCannotGenerate
    assert len(caught) == 1 and "generated every point its search space contains" in caught[0]
    assert _states(study)["PRUNED"] == 1


@pytest.mark.parametrize(
    "reason, text",
    [
        (optquest.OQTerminationReasonException, "an internal engine exception"),
        (optquest.OQTerminationReasonInfeasible, "infeasible"),
        (optquest.OQTerminationReasonRunning, "an error inside the engine"),
        (optquest.OQTerminationReasonNotStarted, "an error inside the engine"),
    ],
)
def test_an_engine_failure_ends_the_study_with_a_warning(monkeypatch, reason, text):
    """An exception in the engine or an infeasible model is not a normal end of the search: the
    sampler prunes the trial, stops the study, and warns once with the reason. So is an invalid
    solution from a search that has not ended (Running, NotStarted): the engine failed."""
    sampler = _sphere_sampler()

    class _NoSolution:
        def is_valid(self):
            return False

        def is_evaluated(self):
            return False

    monkeypatch.setattr(sampler._model, "create_solution", lambda: _NoSolution())
    monkeypatch.setattr(sampler._model, "get_termination_reason", lambda: reason)
    study = optuna.create_study(sampler=sampler)
    caught = _optimize(study, sphere, 3)
    assert _states(study) == {"PRUNED": 1}
    assert len(caught) == 1 and "stopped early" in caught[0] and text in caught[0]


@pytest.mark.parametrize("cmd", ["CONSTRAINT", "EXPRESSION"])
@pytest.mark.parametrize("char", ["\u00a0", "\u00d7", "\u00b2", "\u00fc"])
def test_an_engine_error_that_repeats_a_character_outside_ascii_is_shown(cmd, char):
    """The engine repeats the text of a constraint that it refuses, with one byte for a
    character outside ASCII (a no-break space pasted from a web page, a multiplication sign),
    and the binding cannot decode that byte: the engine's message must still reach the caller."""
    text = f"x {char} 2 > 0.1" if cmd == "CONSTRAINT" else f"x {char} 2"
    with pytest.raises(RuntimeError, match="(?i)error.*(constraint|expression|token)"):
        OptQuestSampler(
            statements=_with_output(X_STMT, {"cmd": cmd, "name": "c1", "expression": text}),
            license="",
        )


def test_the_last_error_of_the_engine_is_text_when_the_binding_cannot_decode_it():
    class _Undecodable:
        def get_last_error(self):
            return b"ERROR Error Token = x \xa0> 0.1".decode("utf-8")

    assert _variable_codec.last_error(_Undecodable()) == "ERROR Error Token = x \ufffd> 0.1"
    module_ = importlib.import_module(f"{_package}.OptQuestSampler")
    assert module_._engine_error(_Undecodable()) == "ERROR Error Token = x \ufffd> 0.1"


def test_a_java_error_inside_the_engine_ends_the_study_with_a_warning():
    """On this grid at -2**50 the 10.2 engine fails when it lists the values (an index out of
    bounds). It then gives an invalid solution, leaves the termination reason at Running and
    raises nothing: the sampler warns with the engine's error, and does not end the study as
    if the search were done."""
    low = -(2.0**50)
    stmt = {"cmd": "DISCRETE", "name": "x", "min": low, "max": low + 0.4, "step": 0.05}
    sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    caught = _optimize(study, sampler.wrap(lambda p: {"out": p["x"]}), 5)
    assert _states(study) == {"PRUNED": 1}
    assert sampler._model.get_termination_reason() == optquest.OQTerminationReasonRunning
    assert len(caught) == 1 and "an error inside the engine" in caught[0]
    assert "ArrayIndexOutOfBounds" in caught[0] and "\tat " not in caught[0]


def test_model_mode_refuses_before_it_sets_a_seed_or_a_key_on_the_model():
    """Every refusal comes before the sampler sets seed= or license= on the caller's model, and
    before it initializes the model, so the model can still be changed."""
    cases = []
    model = _xmodel()
    model.set_sample_method(1)
    cases.append((model, NotImplementedError))
    model = _xmodel()
    model.suggested_solutions_only()
    cases.append((model, NotImplementedError))
    model = _xmodel()
    model.add_integer_variable("v", 0.5, 5.5)
    cases.append((model, ValueError))
    model = _xmodel()
    model.add_integer_variable("v", 0, 2**32 - 1)
    cases.append((model, ValueError))
    model = _xmodel()
    model.add_output_variable("b")
    model.add_minimize_objective("both", "out + b")
    cases.append((model, NotImplementedError))
    for model, error in cases:
        with pytest.raises(error):
            OptQuestSampler(model=model, seed=7, license="K-1")
        described = model.describe()
        assert "RANDOMSEED" not in described and "K-1" not in described
        assert model.add_output_variable("later")  # not initialized


def test_seeds_that_give_the_engine_one_seed_agree_and_an_empty_key_is_no_key(monkeypatch):
    """seed=, a RANDOMSEED statement and a seed set on the model are compared as the seeds that
    the engine gets. The engine ignores a seed below 1 that the caller set on the model, so it
    is not a seed; and a LICENSE statement with an empty key gives no key."""
    statements = _with_output(X_STMT, {"cmd": "RANDOMSEED", "seed": 0})
    sampler = OptQuestSampler(statements=statements, license="", seed=2**31 - 1)
    assert "RANDOMSEED 2147483647" in sampler._model.describe().splitlines()
    for ignored in (0, -3):
        model = _xmodel()
        model.set_random_seed(ignored)
        sampler = OptQuestSampler(model=model, seed=5)
        assert "RANDOMSEED 5" in sampler._model.describe().splitlines()
    keys = []

    def refuse(model, key):
        keys.append(key)
        return False

    monkeypatch.setattr(optquest.OptQuestModel, "set_license", refuse)
    with pytest.raises(RuntimeError, match="refused the license key"):
        OptQuestSampler(
            statements=_with_output(X_STMT, {"cmd": "LICENSE", "license_id": ""}), license="K-1"
        )
    assert keys == ["K-1"]


def test_a_failed_construction_does_not_bind_the_next_one_to_its_seed_or_key(monkeypatch):
    """A construction that fails when the engine initializes the model leaves the seed= and the
    license= that it set on the caller's model. A later sampler for the model does not take
    them for the caller's own: a retry with another seed works, license="" gives demo mode
    again, and a second sampler on an initialized model reports that."""
    model = optquest.OptQuestModel()
    model.set_license("")
    model.add_continuous_variable("x", 0.0, 1.0)
    model.add_output_variable("out")
    with pytest.raises(RuntimeError, match="objective"):
        OptQuestSampler(model=model, seed=3)  # no objective yet
    model.add_minimize_objective("obj", "out")
    sampler = OptQuestSampler(model=model, seed=4)
    assert "RANDOMSEED 4" in sampler._model.describe().splitlines()
    with pytest.raises(RuntimeError, match="already initialized"):
        OptQuestSampler(model=model, seed=5)
    model = _xmodel()
    monkeypatch.setattr(model, "initialize", lambda: False)
    with pytest.raises(RuntimeError):
        OptQuestSampler(model=model, license="K-1")
    monkeypatch.undo()
    assert "K-1" in model.describe()
    OptQuestSampler(model=model)
    assert "K-1" not in model.describe()


def test_model_mode_refuses_a_compound_objective_before_initializing_the_model():
    model = optquest.OptQuestModel()
    model.set_license("")
    model.add_continuous_variable("x", 0.0, 1.0)
    model.add_output_variable("a")
    model.add_output_variable("b")
    model.add_minimize_objective("obj", "a + b")
    with pytest.raises(NotImplementedError, match="identity pass-through"):
        OptQuestSampler(model=model, seed=3)
    assert "RANDOMSEED" not in model.describe()
    assert model.add_output_variable("total")  # the model can still be changed


def test_license_none_runs_in_demo_mode():
    sampler = OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN], license=None, seed=3)
    study = optuna.create_study(sampler=sampler)
    assert _optimize(study, sphere, 2) == []


# ---------------------------------------------------------------------------------------------
# Values: rows, names given as numbers, shaped outputs and wide grids
# ---------------------------------------------------------------------------------------------


def test_a_function_that_changes_a_row_in_place_does_not_change_the_declared_rows():
    """sim() gets a copy of the selected TUPLE row or GEOLOCATION pair, so a change to it does
    not reach a later trial."""
    statements = _with_output(
        {"cmd": "TUPLE", "name": "t", "tuples": [[1, 2], [3, 4]]},
        {"cmd": "GEOLOCATION", "name": "g", "coords": [[10, 20], [30, 40]]},
        X_STMT,
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    seen = []

    def sim(p):
        seen.append((tuple(p["t"]), tuple(p["g"])))
        p["t"].append(999)
        p["g"].reverse()
        return {"out": p["x"] + p["t"][0]}

    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(sim), 12) == []
    assert len(seen) == 12
    assert {t for t, _ in seen} <= {(1, 2), (3, 4)} and {g for _, g in seen} <= {
        (10, 20),
        (30, 40),
    }


def test_statements_coerce_every_form_and_take_names_given_as_numbers():
    """emit_model() accepts a whole float or a numeric string in a field of either form of
    BUDGETEDSELECTION and ITEMSELECTION, and an item or node name given as a number, which it
    writes as text; the typed model methods take neither, so the sampler converts both."""
    statements = _with_output(
        {
            "cmd": "BUDGETEDSELECTION",
            "name": "b",
            "n": 4.0,
            "budget_max": "10",
            "costs": ["3", 4, 5.0, 6],
        },
        {"cmd": "ITEMSELECTION", "name": "s", "n": "5", "k_min": 1.0, "k_max": "3"},
        {"cmd": "PERMUTATION", "name": "route", "items": [0, 1, 2]},
        {"cmd": "DAG", "name": "g", "nodes": [1, 2, 3], "start": 1, "predecessors": {3: [2]}},
        {"cmd": "PATH", "name": "p", "nodes": [1, 2, 3], "path": {"start": 1, "end": 3}},
        {"cmd": "MULTISTAGESEQUENCE", "name": "ms", "stages": [[1, 2], [3]]},
        {"cmd": "TREE", "name": "tr", "nodes": [1, 2, 3], "root": 1},
        {
            "cmd": "TOPOLOGICALSEQUENCE",
            "name": "ts",
            "tasks": [1, 2, 3],
            "predecessors": {3: [1, 2]},
        },
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    assert {"route@0", "g@1,2", "p@3", "ms@1", "tr@1,2", "ts@3"} <= set(sampler._space)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    caught = _optimize(study, sampler.wrap(lambda p: {"out": float(p["route"]["0"])}), 3)
    assert caught == [] and _states(study) == {"COMPLETE": 3}


SHAPED = [
    X_STMT,
    {"cmd": "OUTPUTVECTOR", "name": "v", "n": 3},
    {"cmd": "OUTPUTMATRIX", "name": "m", "rows": 2, "cols": 2},
    {"cmd": "OUTPUTNDVARIABLE", "name": "a", "dims": [2, 1, 2]},
    {"cmd": "OUTPUT", "name": "out"},
    {"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "out"},
]
SHAPED_VALUES = {"out": 0.0, "v": [1, 2, 3], "m": [[1, 2], [3, 4]], "a": [[[5, 6]], [[7, 8]]]}


def test_shaped_outputs_reach_the_engine_by_element():
    sampler = OptQuestSampler(statements=SHAPED, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)

    def sim(p):
        x = p["x"]
        return {**SHAPED_VALUES, "out": x, "v": [x, 2 * x, 3 * x]}

    assert _optimize(study, sampler.wrap(sim), 2) == []
    solutions = sampler._model.get_all_solutions()
    assert len(solutions) == 2
    for s in solutions:
        x = s.get_value("x")
        assert [s.get_value(f"v@{i}") for i in (1, 2, 3)] == pytest.approx([x, 2 * x, 3 * x])
        assert [s.get_value(k) for k in ("m@1,1", "m@1,2", "m@2,1", "m@2,2")] == [1, 2, 3, 4]
        keys = ("a@1,1,1", "a@1,1,2", "a@2,1,1", "a@2,1,2")
        assert [s.get_value(k) for k in keys] == [5, 6, 7, 8]


@pytest.mark.parametrize(
    "name, value, expected",
    [
        ("v", [1, 2], "3 value"),
        ("v", [1, 2, 3, 4], "3 value"),
        ("v", 5.0, "3 value"),
        ("v", [[1], [2], [3]], "3 value"),
        ("m", [[1, 2], [3, 4], [5, 6]], "2x2 matrix"),
        ("m", [[1, 2], [3]], "2x2 matrix"),
        ("a", [[[5, 6]], [[7, 8]], [[9, 9]]], "2x1x2"),
        ("a", [[[5, 6, 7]], [[7, 8]]], "2x1x2"),
        ("a", [[5, 6], [7, 8]], "2x1x2"),
        ("a", 5.0, "2x1x2"),
        # A dict would be read as its keys, and a set in no order of its own.
        ("v", {1: 5.0, 2: 6.0, 3: 7.0}, "3 value"),
        ("v", {0: 5.0, 1: 6.0, 2: 7.0}.keys(), "3 value"),
        ("v", {9.0, 1.0, 5.0}, "3 value"),
        ("v", frozenset({9.0, 1.0, 5.0}), "3 value"),
        ("m", {0: [1, 2], 1: [3, 4]}, "2x2 matrix"),
        ("m", [{0: 7.0, 1: 8.0}, [3, 4]], "2x2 matrix"),
        ("a", [[{0: 5.0, 1: 6.0}], [[7, 8]]], "2x1x2"),
    ],
)
def test_a_shaped_output_of_another_shape_is_refused(name, value, expected):
    sampler = OptQuestSampler(statements=SHAPED, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    with pytest.raises(ValueError, match=f"output {name!r}: expected .*{expected}"):
        study.optimize(sampler.wrap(lambda p: {**SHAPED_VALUES, name: value}), n_trials=1)


def test_a_dict_or_a_set_as_an_element_of_a_shaped_output_is_refused():
    sampler = OptQuestSampler(statements=SHAPED, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    for element in ({1: 2.0}, {2.0}):
        with pytest.raises(ValueError, match="output 'v@3': expected a number"):
            study.optimize(
                sampler.wrap(lambda p: {**SHAPED_VALUES, "v": [1, 2, element]}), n_trials=1
            )


def test_a_shaped_output_is_read_from_any_sequence_in_its_order():
    """A tuple, a range, a deque and a numpy array hold their elements in their places."""
    import collections

    import numpy

    sampler = OptQuestSampler(statements=SHAPED, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    cases = [
        {"v": (7, 8, 9), "m": numpy.array([[1.0, 2.0], [3.0, 4.0]])},
        {"v": range(7, 10), "m": (collections.deque([1, 2]), [3, 4])},
        {"v": numpy.array([7, 8, 9]), "a": numpy.array([[[5, 6]], [[7, 8]]])},
    ]
    results = iter([{**SHAPED_VALUES, **case} for case in cases])
    assert _optimize(study, sampler.wrap(lambda p: next(results)), len(cases)) == []
    solutions = sampler._model.get_all_solutions()
    assert len(solutions) == len(cases)
    for s in solutions:
        assert [s.get_value(f"v@{i}") for i in (1, 2, 3)] == [7, 8, 9]
        assert [s.get_value(k) for k in ("m@1,1", "m@1,2", "m@2,1", "m@2,2")] == [1, 2, 3, 4]
        keys = ("a@1,1,1", "a@1,1,2", "a@2,1,1", "a@2,1,2")
        assert [s.get_value(k) for k in keys] == [5, 6, 7, 8]


def _raw_and_recorded(sampler, name, objective, n_trials):
    """Run a study; return the engine values that the engine proposed for the parameter `name`
    and the ones that its solutions hold afterwards, each in the order of the iterations."""
    raws = []
    decode = sampler._from_engine[name]

    def spy(raw):
        raws.append(raw)
        return decode(raw)

    sampler._from_engine[name] = spy
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, objective, n_trials) == []
    solutions = sorted(sampler._model.get_all_solutions(), key=lambda s: s.get_iteration())
    return raws[: len(solutions)], [s.get_value(name) for s in solutions]


def test_a_choice_given_twice_keeps_the_index_that_the_engine_proposed():
    """Optuna takes a CategoricalDistribution with a choice given twice. The label of the
    second one has the index of the first, and the sampler must not write that index over the
    one that the engine proposed: the engine would hold a point that it did not propose."""
    space = {"c": D.CategoricalDistribution(["relu", "relu", "tanh"])}
    space["x"] = D.FloatDistribution(0.0, 1.0)
    sampler = OptQuestSampler(search_space=space, directions=[MIN], license="", seed=3)

    def objective(trial):
        c = trial.suggest_categorical("c", ["relu", "relu", "tanh"])
        return trial.suggest_float("x", 0.0, 1.0) + (c == "tanh")

    raws, recorded = _raw_and_recorded(sampler, "c", objective, 9)
    assert 2.0 in raws and recorded == raws


@pytest.mark.parametrize("mode", ["statements", "model"])
def test_two_equal_rows_of_a_tuple_keep_the_index_that_the_engine_proposed(mode):
    rows = [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]]
    if mode == "statements":
        stmt = {"cmd": "TUPLE", "name": "t", "tuples": rows}
        sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
    else:
        model = optquest.OptQuestModel()
        model.set_license("")
        model.add_tuple_variable("t", rows)
        model.add_output_variable("out")
        model.add_minimize_objective("obj", "out")
        sampler = OptQuestSampler(model=model, seed=3)
    raws, recorded = _raw_and_recorded(sampler, "t", sampler.wrap(lambda p: {"out": p["t"][0]}), 5)
    assert sorted(raws) == [1.0, 2.0, 3.0] and recorded == raws


def test_a_choice_that_the_objective_replaces_is_written_back():
    """A hand-written objective can suggest a parameter from other choices: Optuna then draws
    a value itself, and that value goes to the engine in place of the engine's."""
    space = {"c": D.CategoricalDistribution(["a", "b", "c"]), "x": D.FloatDistribution(0.0, 1.0)}
    sampler = OptQuestSampler(search_space=space, directions=[MIN], license="", seed=3)

    def objective(trial):
        trial.suggest_categorical("c", ["c"])  # always the third choice
        return trial.suggest_float("x", 0.0, 1.0)

    study = optuna.create_study(sampler=sampler)
    _optimize(study, objective, 6)
    assert {s.get_value("c") for s in sampler._model.get_all_solutions()} == {3.0}


def test_a_plain_grid_wider_than_the_engines_is_searched_whole():
    """The engine's grid holds at most 2**31 - 1 steps, and it would propose only the start of
    a wider IntDistribution or stepped FloatDistribution. The sampler builds such a parameter
    as a CONTINUOUS engine variable and moves each value to the nearest value of the grid."""
    space = {
        "k": D.IntDistribution(0, 2**32 - 1),
        "s": D.IntDistribution(0, 2**36, step=16),
        "f": D.FloatDistribution(0.0, 2.0**33, step=2.0),
    }
    sampler = OptQuestSampler(search_space=space, directions=[MAX], license="", seed=3)
    described = sampler._model.describe().splitlines()
    assert {line.split()[1] for line in described if line.startswith("CONTINUOUS")} == {
        "k",
        "s",
        "f",
    }
    study = optuna.create_study(sampler=sampler, direction="maximize")

    def objective(trial):
        k = trial.suggest_int("k", 0, 2**32 - 1)
        s = trial.suggest_int("s", 0, 2**36, step=16)
        f = trial.suggest_float("f", 0.0, 2.0**33, step=2.0)
        return k + s + f

    assert _optimize(study, objective, 20) == []
    params = [t.params for t in study.trials]
    assert max(p["k"] for p in params) > 2**31 - 1
    assert max(p["s"] for p in params) > 16 * (2**31 - 1)
    assert max(p["f"] for p in params) > 2.0 * (2**31 - 1)
    assert all(type(p["k"]) is int and p["s"] % 16 == 0 and p["f"] % 2 == 0 for p in params)


def test_a_built_model_with_a_grid_wider_than_the_engines_is_refused():
    model = _xmodel()
    model.add_integer_variable("k", 0, 2**32 - 1)
    with pytest.raises(ValueError, match=r"2\*\*31"):
        OptQuestSampler(model=model)
    assert model.add_output_variable("later")  # not initialized


# ---------------------------------------------------------------------------------------------
# External points: parameters with one value, and outputs of another model
# ---------------------------------------------------------------------------------------------


def test_an_external_point_gets_the_value_of_a_parameter_with_one_value():
    """A parameter whose distribution holds one value (the diagonal of a DAG, an INTEGER with
    min == max) can be left out of an external point: the sampler sends that value."""
    statements = _with_output(
        {"cmd": "DAG", "name": "g", "nodes": ["a", "b"]},
        {"cmd": "INTEGER", "name": "k", "min": 5, "max": 5},
        X_STMT,
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    study.enqueue_trial({"g@a,b": 1, "g@b,a": 0, "x": 0.5})

    def objective(trial):
        x = trial.suggest_float("x", -1.0, 1.0)
        return x + trial.suggest_int("g@a,b", 0, 1) + trial.suggest_int("g@b,a", 0, 1)

    assert _optimize(study, objective, 3) == []
    evaluated = [s for s in sampler._model.get_all_solutions() if s.is_evaluated()]
    assert len(evaluated) == 1
    assert [evaluated[0].get_value(k) for k in ("k", "g@a,a", "g@a,b", "x")] == [5, 0, 1, 0.5]


def test_earlier_wrap_trials_without_a_newly_declared_output_are_not_sent():
    """A wrap() trial records the outputs of its own model. A model that declares another output
    would get 0.0 for it, so such an earlier trial is not sent, with one warning."""
    first = OptQuestSampler(statements=_with_output(X_STMT), license="", seed=3)
    study = optuna.create_study(sampler=first, directions=first.directions)
    assert _optimize(study, first.wrap(lambda p: {"out": p["x"]}), 4) == []
    statements = _with_output(
        X_STMT,
        {"cmd": "OUTPUT", "name": "extra"},
        {"cmd": "CONSTRAINT", "name": "c", "expression": "extra >= 1"},
    )
    second = OptQuestSampler(statements=statements, license="", seed=3)
    study.sampler = second
    caught = _optimize(study, second.wrap(lambda p: {"out": p["x"], "extra": 2.0}), 2)
    assert len(caught) == 1 and "['extra']" in caught[0]
    assert len(second._model.get_all_solutions()) == 2  # the four earlier trials were not sent


@pytest.mark.parametrize(
    "variable, low, high",
    [
        ({"cmd": "BINARY", "name": "v"}, 0.2, 0.8),
        ({"cmd": "DESIGN", "name": "v", "num": 5}, 1.5, 3.5),
    ],
)
def test_an_exclusive_range_on_a_binary_or_design_variable_is_the_engines_to_refuse(
    variable, low, high
):
    """The engine refuses an exclusive range on a BINARY or DESIGN variable. The error names the
    statement as it was given."""
    statements = _with_output(variable, {"cmd": "EXCLUSIVE", "name": "v", "min": low, "max": high})
    with pytest.raises(RuntimeError, match="binary or design variable") as raised:
        OptQuestSampler(statements=statements, license="")
    assert f"'min': {low!r}, 'max': {high!r}" in str(raised.value)


# ---------------------------------------------------------------------------------------------
# Fourth review: names, bounds, grids, excluded ranges and external trials
# ---------------------------------------------------------------------------------------------


def _model_without_objective():
    model = optquest.OptQuestModel()
    model.add_continuous_variable("x", 0.0, 1.0)
    model.add_output_variable("out")
    return model


def test_a_failed_construction_keeps_the_callers_own_key_and_seed(monkeypatch):
    """A failed construction records only the values that it changed on the caller's model: a
    key or a seed that the model already had stays the caller's."""
    model = _model_without_objective()
    model.set_license("KEY-123")
    monkeypatch.setattr(model, "initialize", lambda: False)
    with pytest.raises(RuntimeError):
        OptQuestSampler(model=model, license="KEY-123")
    model.add_minimize_objective("obj", "out")
    monkeypatch.setattr(model, "initialize", lambda: True)  # no check of a key that is not one
    OptQuestSampler(model=model)
    assert "KEY-123" in model.describe()
    monkeypatch.undo()
    model = _model_without_objective()
    model.set_license("")
    model.set_random_seed(5)
    with pytest.raises(RuntimeError, match="objective"):
        OptQuestSampler(model=model, seed=5)
    model.add_minimize_objective("obj", "out")
    with pytest.raises(ValueError, match=r"the seed set on the model \(5\)"):
        OptQuestSampler(model=model, seed=6)


@pytest.mark.parametrize(
    "stmt",
    [
        {"cmd": "CONTINUOUS", "name": "変数", "min": 0.0, "max": 1.0},
        {"cmd": "CONTINUOUS", "name": "café", "min": 0.0, "max": 1.0},
        {"cmd": "PERMUTATION", "name": "route", "items": ["Zürich", "Genève", "Bern"]},
        {"cmd": "DAG", "name": "g", "nodes": ["a", "b", "ç"]},
        {"cmd": "OUTPUT", "name": "出力"},
    ],
)
def test_a_name_outside_ascii_is_refused(stmt):
    """The engine gives back NaN for a name with a character outside ASCII, so the sampler
    refuses such a name rather than run on values that are not the engine's."""
    with pytest.raises(ValueError, match="ASCII"):
        OptQuestSampler(statements=_with_output(X_STMT, stmt), license="")


def test_an_objective_and_a_constraint_can_have_a_name_outside_ascii():
    """The engine's reply holds the values of the variables, outputs and expressions only."""
    statements = [
        X_STMT,
        {"cmd": "OUTPUT", "name": "out"},
        {"cmd": "CONSTRAINT", "name": "上限", "expression": "x <= 0.9"},
        {"cmd": "OBJECTIVE", "name": "目的", "direction": "min", "expression": "out"},
    ]
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": p["x"]}), 4) == []
    assert _states(study) == {"COMPLETE": 4}


def test_model_mode_refuses_a_name_outside_ascii_before_initializing_the_model():
    model = _xmodel()
    model.add_continuous_variable("café", 0.0, 1.0)
    with pytest.raises(ValueError, match="ASCII"):
        OptQuestSampler(model=model, seed=3)
    assert "RANDOMSEED" not in model.describe()
    assert model.add_output_variable("later")  # not initialized


def test_plain_mode_takes_a_name_outside_ascii():
    """In plain mode the sampler gives such a name an internal engine name."""
    sampler = OptQuestSampler(
        search_space={"変数": D.FloatDistribution(0.0, 1.0)}, directions=[MIN], license="", seed=3
    )
    study = optuna.create_study(sampler=sampler)
    assert _optimize(study, lambda t: t.suggest_float("変数", 0.0, 1.0), 3) == []
    assert _states(study) == {"COMPLETE": 3}


def test_predecessors_given_as_a_set_are_built_as_names():
    """emit_model() writes a set of predecessors as names, and so does the sampler."""
    statements = _with_output(
        {
            "cmd": "TOPOLOGICALSEQUENCE",
            "name": "ts",
            "tasks": ["a", "b", "c", "d"],
            "predecessors": {"c": {"a", "b"}, "d": frozenset({"a"})},
        }
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    line = [s for s in sampler._model.describe().splitlines() if s.startswith("TOPOLOGICAL")]
    assert len(line) == 1 and "{" not in line[0]
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    seen = []

    def sim(p):
        seen.append(dict(p["ts"]))
        return {"out": float(p["ts"]["c"])}

    assert _optimize(study, sampler.wrap(sim), 4) == []
    assert seen and all(r["a"] < r["c"] and r["b"] < r["c"] and r["a"] < r["d"] for r in seen)


@pytest.mark.parametrize(
    "stmt, field",
    [
        ({"cmd": "TUPLE", "name": "t", "tuples": [[1, 2, 3], [4, 5]]}, "tuples"),
        ({"cmd": "GEOLOCATION", "name": "g", "coords": [[1, 2], [3]]}, "coords"),
        (
            {"cmd": "SPHERICALREGION", "name": "r", "vertices": [[10, 10], [10, 40], [-10]]},
            "vertices",
        ),
        ({"cmd": "REGION", "name": "r", "vertices": [[0, 0], [4, 0, 1], [0, 3, 1]]}, "vertices"),
        (
            {
                "cmd": "LOWRANKCORRECTION",
                "name": "m",
                "initial": [[1, 2, 3], [4, 5]],
                "rank": 1,
                "min": 0.0,
                "max": 9.0,
            },
            "initial",
        ),
        ({"cmd": "GEOLOCATION", "name": "g", "coords": [[1, 2, 3], [4, 5, 6]]}, "coords"),
        ({"cmd": "REGION", "name": "r", "vertices": [[0], [4], [2]]}, "vertices"),
    ],
)
def test_table_rows_that_the_command_does_not_take_are_refused(stmt, field):
    """Rows of different lengths (the typed method fills a short row with zeros), a GEOLOCATION
    row that is not a pair, and a REGION row with fewer than two values."""
    with pytest.raises(ValueError, match=f"{stmt['cmd']} '{stmt['name']}'.*{field}"):
        OptQuestSampler(statements=_with_output(stmt), license="")


def test_region_rows_with_a_third_value_are_taken():
    """The engine reads the first two values of each row as the vertex."""
    stmt = {"cmd": "REGION", "name": "r", "vertices": [[0, 0, 5], [4, 0, 5], [0, 3, 5]]}
    sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    seen = []

    def sim(p):
        seen.append(p["r"])
        return {"out": p["r"]["x"] + p["r"]["y"]}

    assert _optimize(study, sampler.wrap(sim), 4) == []
    assert seen and all(3 * r["x"] + 4 * r["y"] <= 12 + 1e-9 for r in seen)


def test_an_assignment_with_a_list_of_the_wrong_length_is_named():
    stmt = {
        "cmd": "ASSIGNMENT",
        "name": "as",
        "m": 3,
        "n": 2,
        "must_assign": True,
        "bin_cap_min": [0, 0, 0],
        "bin_cap_max": [3, 3, 3],
    }
    with pytest.raises(ValueError, match="ASSIGNMENT 'as'.*one for each bin"):
        OptQuestSampler(statements=_with_output(stmt), license="")


def test_an_assignment_with_one_must_assign_and_caps_for_each_bin():
    """emit_model() accepts one must_assign for every item with a cap list for each bin, and the
    sampler builds that model in the per-item form of the typed method."""
    stmt = {
        "cmd": "ASSIGNMENT",
        "name": "as",
        "m": 3,
        "n": 2,
        "must_assign": True,
        "bin_cap_min": [0, 1],
        "bin_cap_max": [2, 2],
    }
    sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    seen = []

    def sim(p):
        seen.append(list(p["as"]))
        return {"out": float(sum(p["as"]))}

    assert _optimize(study, sampler.wrap(sim), 5) == []
    for bins in seen:
        assert all(b in (1, 2) for b in bins)  # every item is in a bin
        assert bins.count(2) >= 1  # bin 2 holds at least one item


def test_two_excluded_ranges_that_meet_after_fitting_are_merged():
    """Each edge moves outward to a value of the variable: (3.5, 6.5) and (6.5, 9.5) become
    (3, 7) and (6, 10), and the engine would move a value of one into the other. The sampler
    builds them as one range, (3, 10), which excludes the same values."""
    statements = _with_output(
        {"cmd": "INTEGER", "name": "v", "min": 0, "max": 20},
        {"cmd": "EXCLUSIVE", "name": "v", "min": 3.5, "max": 6.5},
        {"cmd": "EXCLUSIVE", "name": "v", "min": 6.5, "max": 9.5},
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    exclusive = [s for s in sampler._model.describe().splitlines() if s.startswith("EXCLUSIVE")]
    assert exclusive == ["EXCLUSIVE v 3 10"]
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    _optimize(study, sampler.wrap(lambda p: {"out": float(abs(p["v"] - 6))}), 30)
    values = {t.params["v"] for t in study.trials if t.params}
    assert values and not values & set(range(4, 10))


def test_model_mode_refuses_excluded_ranges_that_overlap():
    model = _xmodel()
    model.add_integer_variable("v", 0, 20)
    model.add_variable_exclusive_range("v", 3.0, 7.0)
    model.add_variable_exclusive_range("v", 6.0, 10.0)
    with pytest.raises(ValueError, match="overlap"):
        OptQuestSampler(model=model)
    assert model.add_output_variable("later")  # not initialized


def test_two_objectives_on_one_output_need_one_value_from_a_hand_written_objective():
    """The engine holds one value for each output, so two objectives that pass through the same
    output cannot get different values from a hand-written objective."""
    statements = [
        X_STMT,
        {"cmd": "OUTPUT", "name": "o"},
        {"cmd": "OBJECTIVE", "name": "lo", "direction": "min", "expression": "o"},
        {"cmd": "OBJECTIVE", "name": "hi", "direction": "max", "expression": "o"},
    ]
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, lambda t: (t.suggest_float("x", -1.0, 1.0),) * 2, 2) == []
    with pytest.raises(ValueError, match="same output"):
        study.optimize(lambda t: (v := t.suggest_float("x", -1.0, 1.0), 1.0 - v), n_trials=1)


def test_a_trial_added_during_the_run_without_every_output_does_not_fail_the_run():
    """A trial added with study.add_trial() during the run, on a model with an output that no
    objective passes through, is not sent, with one warning; the next trial runs."""
    sampler = OptQuestSampler(statements=CONSTRAINED, license="", seed=5)
    study = optuna.create_study(sampler=sampler)

    def sim(p):
        return {"score": p["x"] ** 2 + p["y"] ** 2, "budget": p["x"] + p["y"]}

    assert _optimize(study, sampler.wrap(sim), 3) == []
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": 1.0, "y": 1.0},
            distributions={
                "x": D.FloatDistribution(-5.0, 10.0),
                "y": D.FloatDistribution(-5.0, 10.0),
            },
            value=2.0,
        )
    )
    caught = _optimize(study, sampler.wrap(sim), 2)
    assert len(caught) == 1 and "passes through" in caught[0]
    assert "added to the study during this run" in caught[0]
    assert _states(study) == {"COMPLETE": 6}
    assert len(sampler._model.get_all_solutions()) == 5


def test_earlier_wrap_trials_with_an_output_that_the_model_dropped_are_sent():
    """An output that the model no longer declares is left out, and the trial is sent with the
    outputs that the model declares."""
    statements = _with_output(X_STMT, {"cmd": "OUTPUT", "name": "extra"})
    first = OptQuestSampler(statements=statements, license="", seed=3)
    study = optuna.create_study(sampler=first, directions=first.directions)
    assert _optimize(study, first.wrap(lambda p: {"out": p["x"], "extra": 1.0}), 4) == []
    second = OptQuestSampler(statements=_with_output(X_STMT), license="", seed=3)
    study.sampler = second
    assert _optimize(study, second.wrap(lambda p: {"out": p["x"]}), 2) == []
    assert len(second._model.get_all_solutions()) == 6


def test_a_second_study_with_the_same_name_and_an_enqueued_trial_is_refused():
    """A fully enqueued trial does not ask the sampler for its parameters, so after_trial()
    refuses a trial whose number this sampler already finished."""
    sampler = _sphere_sampler()
    first = optuna.create_study(study_name="nb", sampler=sampler)
    assert _optimize(first, sphere, 3) == []
    second = optuna.create_study(study_name="nb", sampler=sampler)
    second.enqueue_trial({"x": 1.0, "y": 1.0})
    with pytest.raises(RuntimeError, match="Create a new OptQuestSampler"):
        second.optimize(sphere, n_trials=1)
    assert sampler._model.get_number_of_completed_iterations() == 3


def test_the_name_check_refuses_a_second_study_on_its_own():
    """A study with another name is refused also before this sampler finished any trial."""
    sampler = _sphere_sampler()
    first = optuna.create_study(study_name="one", sampler=sampler)
    trial = first.ask()
    sphere(trial)  # the sampler sees the study "one"; trial 0 is not finished
    second = optuna.create_study(study_name="two", sampler=sampler)
    with pytest.raises(RuntimeError, match="'two'"):
        sphere(second.ask())


def _ask_and_tell(study, objective, n):
    """Ask and tell n trials (a pruned trial is told as PRUNED); return the warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(n):
            trial = study.ask()
            try:
                value = objective(trial)
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue
            study.tell(trial, value)
    return [str(w.message) for w in caught]


def test_the_warnings_at_the_end_of_the_search_are_given_once(monkeypatch):
    """With ask and tell, every trial after the end of the search asks the engine again."""
    sampler = OptQuestSampler(
        statements=_with_output({"cmd": "CHOICE", "name": "c", "k": 3}), license="", seed=3
    )
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    caught = _ask_and_tell(study, sampler.wrap(lambda p: {"out": float(p["c"])}), 8)
    assert sum("generated every point" in w for w in caught) == 1
    assert _states(study)["PRUNED"] > 1

    class _NoSolution:
        def is_valid(self):
            return False

        def is_evaluated(self):
            return False

    sampler = _sphere_sampler()
    monkeypatch.setattr(sampler._model, "create_solution", lambda: _NoSolution())
    monkeypatch.setattr(
        sampler._model, "get_termination_reason", lambda: optquest.OQTerminationReasonException
    )
    study = optuna.create_study(sampler=sampler)
    caught = _ask_and_tell(study, sphere, 3)
    assert sum("stopped early" in w for w in caught) == 1
    assert _states(study) == {"PRUNED": 3}


def test_the_demo_cap_warning_is_given_once():
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(53):
            trial = study.ask()
            try:
                value = sphere(trial)
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue
            study.tell(trial, value)
    assert _states(study) == {"COMPLETE": 50, "PRUNED": 3}
    assert sum("demo-license cap" in str(w.message) for w in caught) == 1


@pytest.mark.parametrize(
    "space",
    [
        {"x": D.IntDistribution(0, 2**51)},
        {"x": D.FloatDistribution(0.0, 1e16)},
        {"x": D.FloatDistribution(-1e16, 0.0, step=1.0)},
        {"x": D.IntDistribution(0, 2**1024)},
    ],
)
def test_a_plain_bound_beyond_the_engines_is_refused(space):
    with pytest.raises(ValueError, match=r"'x'.*2\*\*50"):
        OptQuestSampler(search_space=space, directions=[MIN], license="")


def test_the_largest_bounds_that_the_engine_takes_are_accepted():
    space = {
        "k": D.IntDistribution(-(2**50), 2**50),
        "f": D.FloatDistribution(-(2.0**50), 2.0**50),
    }
    sampler = OptQuestSampler(search_space=space, directions=[MIN], license="", seed=3)
    study = optuna.create_study(sampler=sampler)
    assert (
        _optimize(
            study,
            lambda t: (
                abs(t.suggest_int("k", -(2**50), 2**50))
                + abs(t.suggest_float("f", -(2.0**50), 2.0**50))
            ),
            3,
        )
        == []
    )


@pytest.mark.parametrize(
    "stmt",
    [
        {"cmd": "CONTINUOUS", "name": "x", "min": 0.0, "max": 1e16},
        {"cmd": "INTEGER", "name": "x", "min": 0, "max": 2**1024},
        {"cmd": "DISCRETE", "name": "x", "min": 0.0, "max": 1e30, "step": 1.0},
        {"cmd": "ENUMERATION", "name": "x", "values": [1.0, 2.0, 1e22]},
        {"cmd": "VECTOR", "name": "x", "size": 2, "min": -1e16, "max": 0.0},
    ],
)
def test_a_statement_bound_beyond_the_engines_is_refused(stmt):
    with pytest.raises(ValueError, match=rf"{stmt['cmd']} 'x'.*2\*\*50"):
        OptQuestSampler(statements=_with_output(stmt), license="")


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.add_continuous_variable("w", float("nan"), 1.0),
        lambda m: m.add_angle_variable("a", float("inf"), float("inf")),
        lambda m: m.add_integer_variable("k", 0, 2**51),
    ],
)
def test_model_mode_refuses_a_bound_that_the_engine_does_not_take(build):
    model = _xmodel()
    build(model)
    with pytest.raises(ValueError, match=r"2\*\*50"):
        OptQuestSampler(model=model)
    assert model.add_output_variable("later")  # not initialized


TWO_VALUE_GRIDS = [
    (10000.0, 10000.0001, 0.0001),
    (1000.0, 1000.00001, 1e-5),
    (123456.789, 123456.79, 0.001),
    (1e7, 1e7 + 0.1, 0.1),
    (1048576.0, 1048576.001, 0.001),
]


@pytest.mark.parametrize("low, high, step", TWO_VALUE_GRIDS)
def test_a_two_value_grid_far_from_zero_is_accepted(low, high, step):
    """high - low of a two-value grid can be a few ulps short of one step far from zero; the
    engine then needs a max a few ulps higher, or it refuses the step. (Optuna's own
    FloatDistribution(123456.789, 123456.79, step=0.001) does not hold its high, so the
    objective makes the call of the registered distribution.)"""
    plain = OptQuestSampler(
        search_space={"x": D.FloatDistribution(low, high, step=step)},
        directions=[MIN],
        license="",
        seed=3,
    )
    study = optuna.create_study(sampler=plain)
    assert _optimize(study, lambda t: plain.suggest_all(t)["x"], 3) == []
    assert {t.params["x"] for t in study.trials if t.params} <= {low, high}
    statements = _with_output(
        {"cmd": "DISCRETE", "name": "x", "min": low, "max": high, "step": step}
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": p["x"]}), 3) == []


@pytest.mark.parametrize(
    "low, high, step",
    [
        (-1.001, -0.001, 1.0),
        (-2.3, 0.001, 2.301),
        (-99.1, 0.13, 99.23),
        (-499.9, 0.2, 500.1),
        (-649009.7, 0.57, 649010.27),
    ],
)
@pytest.mark.parametrize("mode", ["plain", "statements", "vector"])
def test_a_two_value_grid_with_a_min_far_below_its_max_is_accepted(mode, low, high, step):
    """high - low is short of one step by a few ulps of the larger bound, here the min: the
    engine gets a max that many ulps higher, and proposes the two values of Optuna's grid."""
    assert D.FloatDistribution(low, high, step=step).high == high  # Optuna keeps the grid
    if mode == "plain":
        sampler = OptQuestSampler(
            search_space={"x": D.FloatDistribution(low, high, step=step)},
            directions=[MIN],
            license="",
            seed=3,
        )
        study = optuna.create_study(sampler=sampler)
        caught = _optimize(study, lambda t: sampler.suggest_all(t)["x"], 4)
    else:
        stmt = {"cmd": "DISCRETE", "name": "x", "min": low, "max": high, "step": step}
        if mode == "vector":
            stmt = {**stmt, "cmd": "DISCRETEVECTOR", "size": 1}
        sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
        study = optuna.create_study(sampler=sampler, directions=sampler.directions)
        caught = _optimize(
            study,
            sampler.wrap(lambda p: {"out": float(p["x"][0] if mode == "vector" else p["x"])}),
            4,
        )
    assert caught == []
    got = {value for trial in study.trials for value in trial.params.values()}
    assert got == {low, high}


def test_a_grid_top_within_a_rounding_error_of_a_grid_value_follows_optuna():
    """The engine keeps a max within 1e-5 of its top grid value as its top, so it gets Optuna's
    top: the top of the grid is 200.00003, and 200.0000303 is not proposed."""
    statements = _with_output(
        {"cmd": "DISCRETE", "name": "x", "min": 200.0, "max": 200.0000303, "step": 1e-5}
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    assert sampler._space["x"].high == 200.00003
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": -p["x"]}), 6) == []
    values = {t.params["x"] for t in study.trials if t.params}
    assert len(values) == 4 and min(values) == 200.0 and max(values) == 200.00003


NARROW_GRIDS = [
    (4e-7, 7e-7, 1e-8),  # a wavelength in metres
    (0.0, 9e-6, 1e-6),
    (1e-6, 1e-5, 1e-6),  # a learning rate
    (0.0, 9e-6, 9e-6),  # two values: the engine stays on the max next to another variable
    (1e-6, 1.1e-5, 5e-6),  # 1.1e-5 - 1e-6 is one ulp below 1e-5
    (100.0, 100.000009, 1e-6),
]


@pytest.mark.parametrize("low, high, step", NARROW_GRIDS)
@pytest.mark.parametrize("mode", ["plain", "statements", "vector", "matrix", "nd", "model"])
def test_a_grid_narrower_than_the_engines_epsilon_is_refused(mode, low, high, step):
    """The engine takes a min and a max within 1e-5 of each other for one value: it proposes
    the max and the min and ends the search, and next to another variable it stays on the max,
    with no error. The sampler refuses such a grid before it builds or initializes a model."""
    stmt = {"cmd": "DISCRETE", "name": "x", "min": low, "max": high, "step": step}
    match = r"'x'.*for one value.*1e-5"
    if mode == "plain":
        space = {
            "y": D.FloatDistribution(0.0, 1.0),
            "x": D.FloatDistribution(low, high, step=step),
        }
        with pytest.raises(ValueError, match=match):
            OptQuestSampler(search_space=space, directions=[MIN], license="")
    elif mode == "model":
        model = optquest.OptQuestModel()
        model.set_license("")
        model.add_discrete_variable("x", low, high, step)
        model.add_output_variable("out")
        model.add_minimize_objective("obj", "out")
        with pytest.raises(ValueError, match=match):
            OptQuestSampler(model=model)
        assert model.add_output_variable("later")  # not initialized
    else:
        shape = {
            "statements": {},
            "vector": {"cmd": "DISCRETEVECTOR", "size": 2},
            "matrix": {"cmd": "DISCRETEMATRIX", "rows": 2, "cols": 2},
            "nd": {"cmd": "DISCRETENDVARIABLE", "dims": [2, 2]},
        }[mode]
        with pytest.raises(ValueError, match=match):
            OptQuestSampler(statements=_with_output({**stmt, **shape}), license="")


@pytest.mark.parametrize(
    "low, high, step", [(0.0, 1e-5, 5e-6), (0.0, 1.2e-5, 4e-6), (1000.0, 1000.00001, 1e-5)]
)
def test_the_refusal_of_a_narrow_grid_is_the_engines_own_test(low, high, step):
    """These grids are on the other side of the engine's test, in the same arithmetic
    (1000.00001 - 1e-5 is 1000.0, not below it), so the sampler takes them. Alone in a model,
    the engine proposes each of their values. (Next to another variable it still takes a value
    within 1e-5 of a bound for the bound: see DISCRETE in the README.)"""
    stmt = {"cmd": "DISCRETE", "name": "x", "min": low, "max": high, "step": step}
    sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": p["x"]}), 8) == []
    values = {t.params["x"] for t in study.trials if t.params}
    assert len(values) == round((high - low) / step) + 1


def test_a_range_without_a_step_can_be_narrower_than_the_engines_epsilon():
    """Only a grid is taken for one value: a CONTINUOUS variable of the same width is searched,
    and so is a stepped distribution with one value."""
    space = {"x": D.FloatDistribution(4e-7, 7e-7), "one": D.FloatDistribution(2.0, 2.0, step=0.5)}
    sampler = OptQuestSampler(search_space=space, directions=[MIN], license="", seed=3)
    study = optuna.create_study(sampler=sampler)
    assert _optimize(study, lambda t: abs(sampler.suggest_all(t)["x"] - 5.5e-7), 12) == []
    values = {t.params["x"] for t in study.trials}
    assert len(values) > 6 and all(4e-7 <= v <= 7e-7 for v in values)
    assert {t.params["one"] for t in study.trials} == {2.0}


def test_statements_must_be_a_list():
    with pytest.raises(TypeError, match="list"):
        OptQuestSampler(
            statements={"cmd": "CONTINUOUS", "name": "x", "min": 0.0, "max": 1.0}, license=""
        )


@pytest.mark.parametrize(
    "stmt",
    [
        {"cmd": "MIXTURE", "name": "m", "components": ["a", "b"], "sum_to": -1.0},
        {"cmd": "CHOICE", "name": "c", "k": 0},
        {"cmd": "DESIGN", "name": "d", "num": 0},
    ],
)
def test_a_statement_that_optuna_cannot_hold_is_named_in_the_error(stmt):
    with pytest.raises(ValueError, match=f"{stmt['cmd']} '{stmt['name']}'"):
        OptQuestSampler(statements=_with_output(stmt), license="")


def test_model_mode_names_the_variable_whose_bounds_are_in_the_wrong_order():
    model = _xmodel()
    model.add_continuous_variable("w", 5.0, 1.0)
    with pytest.raises(ValueError, match="CONTINUOUS 'w'"):
        OptQuestSampler(model=model)


def test_an_objective_that_names_no_output_is_refused_as_such():
    statements = [
        X_STMT,
        {"cmd": "OUTPUT", "name": "out00"},
        {"cmd": "OBJECTIVE", "name": "obj", "direction": "min", "expression": "ot00"},
    ]
    with pytest.raises(ValueError, match="no declared output"):
        OptQuestSampler(statements=statements, license="")


def test_a_policy_sequence_with_a_very_large_feature_value_is_taken():
    """The engine takes a feature value of 1e155, whose square overflows a float."""
    stmt = {
        "cmd": "POLICYSEQUENCE",
        "name": "p",
        "items": ["a", "b", "c", "d"],
        "features": ["f", "g"],
        "matrix": [[1.0, 2.0], [3.0, 1e155], [5.0, 6.0], [7.0, 1.0]],
    }
    sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    caught = _optimize(study, sampler.wrap(lambda p: {"out": float(len(p["p"]))}), 3)
    # The few orders of four items that two features give can all be proposed in 3 trials.
    assert all("generated every point" in w for w in caught)
    assert _states(study)["COMPLETE"] >= 1


def test_a_selection_gives_its_group_when_the_gate_is_on():
    codec = _variable_codec.build({"cmd": "SELECTION", "name": "sel", "vars": ["s1", "s2"]})
    assert codec.decode({"sel": True}) == ["s1", "s2"]
    assert codec.decode({"sel": False}) == []


# ---------------------------------------------------------------------------------------------
# Fifth review: excluded ranges, checked against a model of the engine
# ---------------------------------------------------------------------------------------------
#
# The engine moves a value that falls inside an EXCLUSIVE range to the nearer edge of that range
# (COptQuestVariable.mapToExclusiveRanges: to the lower edge when the value is below the middle
# of the range, else to the upper edge), and it does not look at the other ranges again. So a built model is right when (1) it excludes
# the same values as the ranges that the user gave, (2) every edge that the engine can move a
# value onto is a value that no range excludes, and (3) no two built ranges overlap. The sweep
# below draws variables and ranges at random and checks the three rules for every model that the
# sampler accepts.


def _sweep_values(var):
    cmd = var["cmd"]
    if cmd == "INTEGER":
        return [float(v) for v in range(int(var["min"]), int(var["max"]) + 1)]
    if cmd == "DISCRETE":
        low, top, step = float(var["min"]), float(var["max"]), float(var["step"])
        values, k = [], 0
        while low + k * step <= top + 1e-9 * max(1.0, abs(top)):
            values.append(low + k * step)
            k += 1
        return values
    if cmd == "ENUMERATION":
        return sorted(float(v) for v in var["values"])
    rows = var["tuples"] if cmd == "TUPLE" else var["coords"]
    return [float(i) for i in range(1, len(rows) + 1)]


def _sweep_tol(x):
    return 1e-9 * max(1.0, abs(x))


def _sweep_inside(x, a, b):
    return a + _sweep_tol(x) < x < b - _sweep_tol(x)


def _sweep_excluded(x, ranges):
    return any(_sweep_inside(x, a, b) for a, b in ranges)


def _sweep_problems(values, user, built):
    """What is wrong with the `built` ranges for the `user` ranges, or []."""
    problems = []
    want = [_sweep_excluded(x, user) for x in values]
    got = [_sweep_excluded(x, built) for x in values]
    if want != got:
        problems.append(
            f"excluded {[x for x, e in zip(values, got) if e]} instead of "
            f"{[x for x, e in zip(values, want) if e]}"
        )
    for a, b in built:
        for x in values:
            if not a < x < b:
                continue
            edge = a if x < a + (b - a) / 2.0 else b
            on_value = any(abs(edge - v) <= _sweep_tol(v) for v in values)
            if not on_value or _sweep_excluded(edge, built):
                problems.append(f"the engine moves {x!r} onto {edge!r}")
    spans = sorted(built)
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        if a2 < b1 - _sweep_tol(b1):
            problems.append(f"({a1!r}, {b1!r}) and ({a2!r}, {b2!r}) overlap")
    return problems


def _sweep_variable(rng):
    kind = rng.choice(["INTEGER", "DISCRETE", "DISCRETE", "ENUMERATION", "TUPLE"])
    if kind == "INTEGER":
        low = rng.choice([-3, 0, 5])
        return {"cmd": "INTEGER", "name": "x", "min": low, "max": low + rng.randint(3, 12)}
    if kind == "DISCRETE":
        low, step = rng.choice(
            [
                ("0", "0.1"),
                ("0", "0.25"),
                ("-1.2", "0.05"),
                ("1000", "0.00001"),
                ("0.1", "0.1"),
                ("123.4", "0.01"),
                ("-0.3", "0.1"),
            ]
        )
        n = rng.randint(3, 15)
        top = decimal.Decimal(low) + n * decimal.Decimal(step)
        return {
            "cmd": "DISCRETE",
            "name": "x",
            "min": float(low),
            "max": float(top),
            "step": float(step),
        }
    if kind == "ENUMERATION":
        values = sorted({round(rng.uniform(-5.0, 10.0), 2) for _ in range(rng.randint(3, 8))})
        return {"cmd": "ENUMERATION", "name": "x", "values": values}
    rows = [[float(i), float(2 * i)] for i in range(1, rng.randint(3, 8) + 1)]
    return {"cmd": "TUPLE", "name": "x", "tuples": rows}


def _sweep_edges(rng, values, var):
    """Candidate edges: the values, the points between them, points beyond the values, decimal
    neighbours of grid values (0.6 for 0.6000000000000001) and random points."""
    span = values[-1] - values[0]
    pool = list(values)
    pool += [(a + b) / 2.0 for a, b in zip(values, values[1:])]
    pool += [values[0] - span * f for f in (0.05, 0.3, 1.0)]
    pool += [values[-1] + span * f for f in (0.05, 0.3, 1.0)]
    pool += [float(repr(round(v, 9))) for v in values]
    pool += [rng.uniform(values[0] - span * 0.5, values[-1] + span * 0.5) for _ in range(6)]
    return pool


def _sweep_ranges(rng, values, var):
    pool = _sweep_edges(rng, values, var)
    ranges = []
    for _ in range(rng.randint(1, 3)):
        a, b = sorted(rng.sample(pool, 2))
        if a < b:
            ranges.append((a, b))
    return ranges


def _built_ranges(model):
    out = []
    for line in model.describe().splitlines():
        if line.startswith("EXCLUSIVE "):
            _, _, low, high = line.split()
            out.append((float(low), float(high)))
    return out


def _typed_model(var, ranges):
    model = optquest.OptQuestModel()
    model.set_license("")
    cmd = var["cmd"]
    if cmd == "INTEGER":
        model.add_integer_variable("x", var["min"], var["max"])
    elif cmd == "DISCRETE":
        model.add_discrete_variable("x", var["min"], var["max"], var["step"])
    elif cmd == "ENUMERATION":
        model.add_enumeration_variable("x", var["values"])
    else:
        model.add_tuple_variable("x", var["tuples"])
    for a, b in ranges:
        model.add_variable_exclusive_range("x", a, b)
    model.add_output_variable("out")
    model.add_minimize_objective("obj", "out")
    return model


@pytest.mark.parametrize("mode", ["statements", "model"])
def test_excluded_ranges_are_built_right_for_random_variables_and_ranges(mode):
    import random

    rng = random.Random(20261110 if mode == "statements" else 20261111)
    failures, accepted, refused = [], 0, 0
    for case in range(700):
        var = _sweep_variable(rng)
        values = _sweep_values(var)
        user = _sweep_ranges(rng, values, var)
        if not user:
            continue
        try:
            if mode == "statements":
                stmts = [var] + [
                    {"cmd": "EXCLUSIVE", "name": "x", "min": a, "max": b} for a, b in user
                ]
                sampler = OptQuestSampler(statements=_with_output(*stmts), license="")
            else:
                sampler = OptQuestSampler(model=_typed_model(var, user))
        except ValueError as exc:
            assert "EXCLUSIVE" in str(exc), (var, user, str(exc))
            refused += 1
            if mode == "statements" and not _sweep_refusal_is_one_range_refusal(var, user, exc):
                failures.append((var, user, f"refused, but no one range is: {exc}"))
            continue
        accepted += 1
        built = _built_ranges(sampler._model)
        problems = _sweep_problems(values, user, built)
        if problems:
            failures.append(
                (var["cmd"], var.get("min"), var.get("step"), user, built, problems[:2])
            )
    assert not failures, f"{len(failures)} of {accepted} accepted cases: {failures[:4]}"
    assert accepted > (200 if mode == "statements" else 80), (accepted, refused)


def _sweep_refusal_is_one_range_refusal(var, user, exc):
    """A refusal of several ranges is the refusal of one range: one of them alone, or the one
    range that the sampler merged them into (the error gives it)."""
    match = re.search(r"one range from (\S+) to (\S+);", str(exc))
    candidates = [(float(match.group(1)), float(match.group(2)))] if match else list(user)
    for a, b in candidates:
        stmts = [var, {"cmd": "EXCLUSIVE", "name": "x", "min": a, "max": b}]
        try:
            OptQuestSampler(statements=_with_output(*stmts), license="")
        except ValueError:
            return True
    return False


@pytest.mark.parametrize("mode", ["statements", "model"])
def test_the_engine_keeps_out_of_the_excluded_ranges_that_the_sampler_accepts(mode):
    """The same sweep, run on the engine: no trial gets a value that the user excluded, and
    Optuna discards no value."""
    import random

    rng = random.Random(7 if mode == "statements" else 8)
    checked = 0
    while checked < 30:
        var = _sweep_variable(rng)
        values = _sweep_values(var)
        user = _sweep_ranges(rng, values, var)
        if not user:
            continue
        try:
            if mode == "statements":
                stmts = [var] + [
                    {"cmd": "EXCLUSIVE", "name": "x", "min": a, "max": b} for a, b in user
                ]
                sampler = OptQuestSampler(statements=_with_output(*stmts), license="", seed=3)
            else:
                sampler = OptQuestSampler(model=_typed_model(var, user), seed=3)
        except ValueError:
            continue
        checked += 1
        study = optuna.create_study(sampler=sampler, directions=sampler.directions)
        target = rng.choice(values)

        def sim(p):
            value = p["x"][0] if var["cmd"] == "TUPLE" else p["x"]
            return {"out": float(abs(value - target))}

        caught = _optimize(study, sampler.wrap(sim), 25)
        assert not [w for w in caught if "discarded" in w], (var, user, caught)
        for trial in study.trials:
            if not trial.params:
                continue
            got = trial.params["x"]
            index = float(ast.literal_eval(got)[0]) if var["cmd"] == "TUPLE" else float(got)
            assert not _sweep_excluded(index, user), (var, user, got)


@pytest.mark.parametrize(
    "low, high, step, first, second, kept",
    [
        (0.0, 1.0, 0.1, (0.15, 0.55), (0.6, 0.95), 0.6),
        (-1.2, -0.6, 0.05, (-1.1745, -1.0745), (-1.05, -0.95), -1.05),
        (0.1, 1.6, 0.1, (0.85, 1.451), (1.5, 1.551), 1.5),
    ],
)
def test_excluded_ranges_that_only_touch_keep_the_value_between_them(
    low, high, step, first, second, kept
):
    """An edge moved onto the grid value 0.6000000000000001 touches the edge 0.6 of the next
    range: the two ranges are kept, and the value between them is not excluded."""
    stmts = [{"cmd": "DISCRETE", "name": "x", "min": low, "max": high, "step": step}]
    stmts += [{"cmd": "EXCLUSIVE", "name": "x", "min": a, "max": b} for a, b in (first, second)]
    sampler = OptQuestSampler(statements=_with_output(*stmts), license="")
    built = _built_ranges(sampler._model)
    assert len(built) == 2
    assert built[0][1] == built[1][0]  # one shared edge, inside neither range
    assert not _sweep_excluded(kept, built)


def test_merged_excluded_ranges_are_checked_as_one_range():
    """(-5, 3) and (2, 8) on an INTEGER from 0 to 10 overlap; as one range from -5 to 8, the
    engine would move 0 onto -5, so the sampler refuses them as it refuses that range alone."""
    stmts = [{"cmd": "INTEGER", "name": "x", "min": 0, "max": 10}]
    for pair in ([(-5, 3), (2, 8)], [(-5, 8)]):
        exclusive = [{"cmd": "EXCLUSIVE", "name": "x", "min": a, "max": b} for a, b in pair]
        with pytest.raises(ValueError, match="EXCLUSIVE 'x'.*-5"):
            OptQuestSampler(statements=_with_output(*stmts, *exclusive), license="")


def test_the_one_range_that_model_mode_suggests_for_overlapping_ranges_is_taken():
    model = _typed_model({"cmd": "INTEGER", "min": 0, "max": 10}, [(1.0, 4.0), (3.0, 6.0)])
    with pytest.raises(ValueError, match=r"from 1\.0 to 6\.0") as caught:
        OptQuestSampler(model=model)
    assert "overlap" in str(caught.value)
    OptQuestSampler(model=_typed_model({"cmd": "INTEGER", "min": 0, "max": 10}, [(1.0, 6.0)]))
    model = _typed_model({"cmd": "INTEGER", "min": 0, "max": 10}, [(-5.0, 3.0), (2.0, 8.0)])
    with pytest.raises(ValueError, match="min or max"):
        OptQuestSampler(model=model)


TENTHS = [0.1 * i for i in range(11)]  # 0.30000000000000004, 0.6000000000000001, ...


@pytest.mark.parametrize(
    "values, low, high, fitted",
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], 2.0, 4.000000000000001, (2.0, 5.0)),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 1.999999999999999, 4.0, (1.0, 4.0)),
        (TENTHS, 0.3, 0.8, (0.2, 0.8)),
        (TENTHS, 0.3, 0.7, (0.2, 0.7)),  # 0.7 is just outside 0.7000000000000001: kept
    ],
)
def test_an_excluded_range_with_an_edge_next_to_an_enumeration_value_is_fitted(
    values, low, high, fitted
):
    """An edge one rounding error from a declared value is not that value (0.3 and the
    0.30000000000000004 of 0.1 * 3), and the value is then inside the range: the engine would
    propose the edge, which is not a choice of the Optuna distribution, and the study would stop
    with an error. The edge moves to the next value (statements=) or the range is refused
    (model=), as for any edge that is not a value."""
    var = {"cmd": "ENUMERATION", "name": "x", "values": values}
    stmt = {"cmd": "EXCLUSIVE", "name": "x", "min": low, "max": high}
    sampler = OptQuestSampler(statements=_with_output(var, stmt), license="", seed=3)
    assert _built_ranges(sampler._model) == [fitted]
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": abs(p["x"] - 0.45)}), 15) == []
    got = [t.params["x"] for t in study.trials if t.params]
    assert got and all(x in values and not low < x < high for x in got)
    with pytest.raises(ValueError, match=r"Declare the range from .* which excludes the same"):
        OptQuestSampler(model=_typed_model(var, [(low, high)]))


@pytest.mark.parametrize("low, high", [(2.0, 3.999999999999999), (2.000000000000001, 4.0)])
@pytest.mark.parametrize("mode", ["statements", "model"])
def test_an_excluded_range_with_an_edge_just_outside_an_enumeration_value_is_kept(mode, low, high):
    """An edge one rounding error on the other side of a declared value leaves the value outside
    the range, and the engine proposes the value: the range is taken as it is in both modes."""
    var = {"cmd": "ENUMERATION", "name": "x", "values": [1.0, 2.0, 3.0, 4.0, 5.0]}
    if mode == "statements":
        stmt = {"cmd": "EXCLUSIVE", "name": "x", "min": low, "max": high}
        sampler = OptQuestSampler(statements=_with_output(var, stmt), license="", seed=3)
    else:
        sampler = OptQuestSampler(model=_typed_model(var, [(low, high)]), seed=3)
    assert _built_ranges(sampler._model) == [(low, high)]
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": abs(p["x"] - 2.9)}), 15) == []
    assert sorted(t.params["x"] for t in study.trials if t.params) == [1.0, 2.0, 4.0, 5.0]


@pytest.mark.parametrize(
    "var, low, high, kept",
    [
        ({"cmd": "INTEGER", "min": 0, "max": 10}, 5.0, float("inf"), lambda x: x <= 5),
        ({"cmd": "INTEGER", "min": 0, "max": 10}, float("-inf"), 5.0, lambda x: x >= 5),
        ({"cmd": "DISCRETE", "min": 0.0, "max": 1.0, "step": 0.1}, 0.5, 1e308, lambda x: x <= 0.5),
        (
            {"cmd": "DISCRETE", "min": 0.0, "max": 1.0, "step": 0.1},
            -1e308,
            0.5,
            lambda x: x >= 0.5,
        ),
    ],
)
def test_a_model_with_an_excluded_range_up_to_infinity_is_taken(var, low, high, kept):
    """The engine takes an infinite edge on a built model, and moves every value inside the
    range to the other edge."""
    sampler = OptQuestSampler(model=_typed_model(var, [(low, high)]), seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": abs(p["x"] - 0.3)}), 15) == []
    got = [t.params["x"] for t in study.trials if t.params]
    assert len(got) > 3 and all(kept(x) for x in got)
    if math.isfinite(low) and math.isfinite(high):
        stmts = [{**var, "name": "x"}, {"cmd": "EXCLUSIVE", "name": "x", "min": low, "max": high}]
        built = OptQuestSampler(statements=_with_output(*stmts), license="")
        assert _built_ranges(built._model) == [(low, high)]


@pytest.mark.parametrize("cmd, field", [("TUPLE", "tuples"), ("GEOLOCATION", "coords")])
def test_an_excluded_range_on_the_rows_of_a_tuple_is_checked(cmd, field):
    """The engine excludes the 1-based row index, and proposes the edges of the range."""
    rows = [[float(i), float(i) / 10.0] for i in range(1, 9)]
    var = {"cmd": cmd, "name": "t", field: rows}
    fitted = OptQuestSampler(
        statements=_with_output(var, {"cmd": "EXCLUSIVE", "name": "t", "min": 1.5, "max": 5.5}),
        license="",
    )
    assert _built_ranges(fitted._model) == [(1.0, 6.0)]
    with pytest.raises(ValueError, match=r"EXCLUSIVE 't'.*rows"):
        OptQuestSampler(
            statements=_with_output(var, {"cmd": "EXCLUSIVE", "name": "t", "min": 2, "max": 9}),
            license="",
        )
    study = optuna.create_study(sampler=fitted, directions=fitted.directions)
    seen = []

    def sim(p):
        seen.append(p["t"])
        return {"out": float(p["t"][0])}

    assert _optimize(study, fitted.wrap(sim), 8) == []
    assert seen and all(row[0] not in (2.0, 3.0, 4.0, 5.0) for row in seen)


@pytest.mark.parametrize(
    "stmt, key",
    [
        ({"cmd": "ANGLE", "name": "dir", "min_deg": -180.0, "max_deg": 180.0}, "dir@theta"),
        ({"cmd": "ANGLE", "name": "dir", "min_deg": 320.0, "max_deg": 40.0}, "dir@theta"),
        (
            {
                "cmd": "SPHERICALANGLE",
                "name": "dir",
                "min_theta_deg": -30.0,
                "max_theta_deg": 30.0,
                "min_phi_deg": -180.0,
                "max_phi_deg": 180.0,
            },
            "dir@phi",
        ),
    ],
)
def test_a_constraint_on_an_angle_sees_the_value_that_the_engine_proposed(stmt, key):
    """The sampler gives the engine back the angle in [0, 360), as the engine proposed it, so a
    constraint on the angle is judged on that value and not on the one in the declared range."""
    statements = _with_output(
        stmt, {"cmd": "CONSTRAINT", "name": "c", "expression": f"{key} >= 180"}
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": 1.0}), 20) == []
    solutions = sampler._model.get_all_solutions()
    assert len(solutions) == 20
    for solution in solutions:
        angle = solution.get_value(key)
        assert 0.0 <= angle < 360.0
        assert solution.is_feasible() == (angle >= 180.0 - 1e-9)


@pytest.mark.parametrize(
    "min_deg, max_deg",
    [
        (-180.0, 180.0),
        (320.0, 40.0),
        (360.0, 0.0),
        (0.0, 90.0),
        (0.0, 360.0),
        (-45.0, 45.0),
        (400.0, 30.0),
    ],
)
def test_an_angle_goes_to_the_engine_in_0_to_360_and_comes_back(min_deg, max_deg):
    import random

    codec = _variable_codec.build(
        {"cmd": "ANGLE", "name": "a", "min_deg": min_deg, "max_deg": max_deg}
    )
    dist = codec.params["a@theta"]
    to_engine, from_engine = codec.to_engine["a@theta"], codec.from_engine["a@theta"]
    rng = random.Random(0)
    for value in [dist.low, dist.high] + [rng.uniform(dist.low, dist.high) for _ in range(500)]:
        engine_value = to_engine(value)
        assert 0.0 <= engine_value < 360.0
        turns = (value - engine_value) / 360.0
        assert abs(turns - round(turns)) <= 1e-9, (value, engine_value)
    # The engine proposes the angles of the arc, in [0, 360).
    arc = [
        to_engine(v)
        for v in [dist.low, dist.high] + [rng.uniform(dist.low, dist.high) for _ in range(500)]
    ]
    for engine_value in arc:
        value = from_engine(engine_value)
        assert dist._contains(value), (engine_value, value)
        back = to_engine(value)
        assert math.isclose(back, engine_value, abs_tol=1e-9) or math.isclose(
            abs(back - engine_value), 360.0, abs_tol=1e-9
        )


@pytest.mark.parametrize("enqueue", [False, True])
def test_a_study_with_the_wrong_direction_does_not_bind_the_sampler(enqueue):
    """The error tells the user to create the study with directions=sampler.directions; that
    study then runs, because nothing of the refused study reached the engine."""
    sampler = _sphere_sampler()
    wrong = optuna.create_study(sampler=sampler, direction="maximize")
    if enqueue:
        wrong.enqueue_trial({"x": 1.0, "y": 2.0})  # every parameter: the sampler is not asked
    with pytest.raises(ValueError, match="directions=sampler.directions"):
        wrong.optimize(sphere, n_trials=1)
    right = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(right, sphere, 3) == []
    assert _states(right) == {"COMPLETE": 3}
    assert sampler._model.get_number_of_completed_iterations() == 3


NEAR_ZERO_GRIDS = [
    (1e-6, 0.020001, 0.001),
    (1e-6, 10.000001, 0.5),
    (-0.020001, -1e-6, 0.001),
    (-10.00001, -1e-5, 0.5),
]


@pytest.mark.parametrize("low, high, step", NEAR_ZERO_GRIDS)
@pytest.mark.parametrize("mode", ["plain", "statements", "vector", "model"])
def test_the_engines_zero_next_to_a_grid_near_zero_is_moved_onto_the_grid(mode, low, high, step):
    """Once it has proposed every value of a grid whose min is within 1e-5 above 0 (or whose
    max is within 1e-5 below 0), the engine proposes 0.0, which is outside the grid: its
    epsilon of 1e-5 takes 0.0 for the bound. The sampler moves it onto the bound."""
    if mode == "plain":
        sampler = OptQuestSampler(
            search_space={"x": D.FloatDistribution(low, high, step=step)},
            directions=[MIN],
            license="",
            seed=3,
        )
    elif mode in ("statements", "vector"):
        stmt = {"cmd": "DISCRETE", "name": "x", "min": low, "max": high, "step": step}
        if mode == "vector":
            stmt = {**stmt, "cmd": "DISCRETEVECTOR", "size": 1}
        sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
    else:
        model = optquest.OptQuestModel()
        model.set_license("")
        model.add_discrete_variable("x", low, high, step)
        model.add_output_variable("out")
        model.add_minimize_objective("obj", "out")
        sampler = OptQuestSampler(model=model, seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    if mode == "plain":
        caught = _optimize(study, lambda t: sampler.suggest_all(t)["x"], 30)
    else:
        caught = _optimize(
            study,
            sampler.wrap(lambda p: {"out": float(p["x"][0] if mode == "vector" else p["x"])}),
            30,
        )
    assert not [w for w in caught if "discarded" in w], caught
    for trial in study.trials:
        for value in trial.params.values():
            assert low - 1e-12 <= value <= high + 1e-12


@pytest.mark.parametrize("storage", ["memory", "sqlite"])
def test_a_nan_choice_of_a_plain_categorical_reaches_the_engine(tmp_path, storage):
    """Optuna gives back a NaN choice as another NaN object (from RDB storage, or from the list
    of a suggest_categorical() call): the sampler finds its index as Optuna does."""
    space = {"c": D.CategoricalDistribution([float("nan"), 1.0, 2.0])}
    sampler = OptQuestSampler(search_space=space, directions=[MIN], license="", seed=3)
    url = None if storage == "memory" else f"sqlite:///{tmp_path / 'nan.db'}"
    study = optuna.create_study(sampler=sampler, storage=url)

    def objective(trial):
        choice = trial.suggest_categorical("c", [float("nan"), 1.0, 2.0])
        return -100.0 if math.isnan(choice) else choice

    assert _optimize(study, objective, 6) == []  # three choices: the search ends after three
    solutions = sampler._model.get_all_solutions()
    assert len(solutions) == 3 and all(s.is_feasible() for s in solutions)
    assert min(s.get_value("OQout00") for s in solutions) == -100.0


def test_a_plain_categorical_index_outside_the_choices_is_an_error():
    to_engine, from_engine = importlib.import_module(
        f"{_package}.OptQuestSampler"
    )._index_converters(D.CategoricalDistribution(["a", "b", "c"]), "c")
    assert [from_engine(1.0), from_engine(3.0), from_engine(2.0000000001)] == ["a", "c", "b"]
    assert to_engine("b") == 2.0
    for raw in (0.0, 4.0):
        with pytest.raises(ValueError, match="out of range"):
            from_engine(raw)


def test_a_model_that_a_sampler_initialized_is_refused_by_the_next_sampler():
    """The engine keeps the state of the search in the model: a second sampler would go on with
    the first sampler's search (and its demo-cap count) with no error."""
    model = _xmodel()
    first = OptQuestSampler(model=model, seed=3)
    study = optuna.create_study(sampler=first, directions=first.directions)
    assert _optimize(study, first.wrap(lambda p: {"out": p["x"]}), 3) == []
    for kwargs in ({}, {"seed": 3}, {"seed": 4}, {"license": ""}):
        with pytest.raises(
            RuntimeError, match="already initialized by an earlier OptQuestSampler"
        ):
            OptQuestSampler(model=model, **kwargs)
    built = OptQuestSampler(statements=_with_output(X_STMT), license="")
    with pytest.raises(RuntimeError, match="already initialized by an earlier OptQuestSampler"):
        OptQuestSampler(model=built._model)


def test_a_seed_or_a_key_that_an_initialized_model_already_has_is_not_set_again(monkeypatch):
    """The engine refuses to set a seed or a key on a model that is initialized, also the one
    that the model has; the sampler sets only a value that differs."""
    model = _xmodel()
    model.set_random_seed(5)
    assert model.initialize()
    sampler = OptQuestSampler(model=model, seed=5)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": p["x"]}), 3) == []
    model = _xmodel()
    model.set_license("KEY-123")
    monkeypatch.setattr(model, "initialize", lambda: True)  # no check of a key that is not one
    monkeypatch.setattr(model, "set_license", lambda key: False)  # as on an initialized model
    OptQuestSampler(model=model, license="KEY-123")


PARTIAL_DAG = [
    X_STMT,
    {"cmd": "DAG", "name": "flow", "nodes": ["a", "b", "c"]},
]


def test_a_trial_enqueued_without_every_parameter_of_a_dag_is_warned_once():
    """Optuna's RandomSampler draws each parameter that a partly enqueued trial does not fix,
    one at a time, so a DAG of such a trial can have a cycle."""
    sampler = OptQuestSampler(statements=_with_output(*PARTIAL_DAG), license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    whole = {f"flow@{u},{v}": 0 for u in "abc" for v in "abc" if u != v}
    study.enqueue_trial({"x": 0.5, **whole})  # every parameter: no warning
    caught = _optimize(study, sampler.wrap(lambda p: {"out": p["x"]}), 1)
    assert not [w for w in caught if "enqueued without" in w]
    study.enqueue_trial({"x": 0.5})
    study.enqueue_trial({"x": 0.25, "flow@a,b": 1})
    caught = _optimize(study, sampler.wrap(lambda p: {"out": p["x"]}), 3)
    partial = [w for w in caught if "enqueued without every parameter of 'flow'" in w]
    assert len(partial) == 1 and "DAG" in partial[0]


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize(
    "fixed, warned",
    [
        ({"p@a": 1}, True),  # one rank of three: the engine's other ranks can repeat it
        ({"p@a": 1, "p@b": 2}, True),
        ({"p@a": 1, "p@b": 2, "p@c": 3}, False),  # the whole variable
        ({"x": 0.5}, False),  # a variable of its own
        ({"x": 0.5, "p@a": 1, "p@b": 2, "p@c": 3}, False),
    ],
)
def test_a_wrapper_that_fixes_part_of_a_permutation_is_warned_once(fixed, warned):
    """optuna.samplers.PartialFixedSampler takes the parameters that it fixes out of the search
    space, and its value replaces the engine's for those parameters only: with one rank of a
    PERMUTATION fixed, the ranks that reach the objective can repeat."""
    stmts = _with_output(X_STMT, {"cmd": "PERMUTATION", "name": "p", "items": ["a", "b", "c"]})
    sampler = OptQuestSampler(statements=stmts, license="", seed=3)
    wrapper = optuna.samplers.PartialFixedSampler(fixed, sampler)
    study = optuna.create_study(sampler=wrapper, directions=sampler.directions)
    caught = _optimize(study, sampler.wrap(lambda p: {"out": p["x"]}), 4)
    partial = [w for w in caught if "PartialFixedSampler" in w]
    assert len(caught) == len(partial) == (1 if warned else 0)
    if warned:
        assert "'p' (PERMUTATION)" in partial[0] and "not 'p@a'" in partial[0]


def test_statements_with_a_misspelled_predecessor_or_path_field_are_refused():
    """A name that the engine would ignore, and a field of `path` that emit_model() drops,
    would build another model than the one declared."""
    topo = {"cmd": "TOPOLOGICALSEQUENCE", "name": "ts", "tasks": ["a", "b", "c"]}
    with pytest.raises(ValueError, match="TOPOLOGICALSEQUENCE 'ts': predecessors names 'aa'"):
        OptQuestSampler(
            statements=_with_output({**topo, "predecessors": {"b": ["aa"]}}), license=""
        )
    path = {"cmd": "PATH", "name": "p", "nodes": ["a", "b", "c", "d"]}
    with pytest.raises(ValueError, match="PATH 'p': path has no field 'start_node'"):
        OptQuestSampler(
            statements=_with_output({**path, "path": {"start_node": "a", "end_node": "d"}}),
            license="",
        )
    sampler = OptQuestSampler(
        statements=_with_output(
            {**topo, "predecessors": {"b": ["a"], "c": {"b"}}},
            {**path, "path": {"start": "a", "end": "d", "required": ["b"]}},
        ),
        license="",
        seed=3,
    )
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    seen = []

    def sim(p):
        seen.append(p)
        return {"out": 0.0}

    assert _optimize(study, sampler.wrap(sim), 3) == []  # the space holds three points
    assert len(seen) == 3
    for p in seen:
        assert p["ts"]["a"] < p["ts"]["b"] < p["ts"]["c"]
        assert p["p"]["a"] == 1 and p["p"]["b"] > 1 and p["p"]["d"] == max(p["p"].values())


def test_a_partly_enqueued_trial_of_plain_parameters_is_not_warned():
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    study.enqueue_trial({"x": 1.0})
    assert _optimize(study, sphere, 2) == []


def test_a_name_given_as_a_number_gets_the_check_of_a_name():
    """emit_model() checks a name only when it is a str; the sampler gives it the str first."""
    with pytest.raises(optquest.EmitError, match="starts with a digit"):
        OptQuestSampler(
            statements=_with_output({"cmd": "CONTINUOUS", "name": 5, "min": 0.0, "max": 1.0}),
            license="",
        )


def test_a_file_or_an_expression_that_is_not_a_str_is_taken_as_text(tmp_path, monkeypatch):
    import pathlib

    # A file name with no directory: the binding refuses whitespace in a file, and the path of
    # tmp_path can have some (a user name with a space).
    monkeypatch.chdir(tmp_path)
    statements = _with_output(
        X_STMT,
        {"cmd": "EXPRESSION", "name": "five", "expression": 5},
        {"cmd": "LOGSETUP", "file": pathlib.Path("setup.log")},
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": p["x"]}), 2) == []


@pytest.mark.parametrize(
    "stmt, field",
    [
        (
            {
                "cmd": "ASSIGNMENT",
                "name": "a",
                "m": 2,
                "n": 2,
                "must_assign": False,
                "bin_cap_min": 0,
                "bin_cap_max": 2**32 + 1,
            },
            "bin_cap_max",
        ),
        (
            {
                "cmd": "ASSIGNMENT",
                "name": "a",
                "m": 2,
                "n": 2,
                "must_assign": False,
                "bin_cap_min": [0, 0],
                "bin_cap_max": [2**31, 2],
            },
            "bin_cap_max",
        ),
        ({"cmd": "STOPTIME", "seconds": 2**31}, "seconds"),
        # A count that stays small if the check is lost (its low 32 bits are 3): one of 2**31
        # elements would take all the memory of the machine.
        ({"cmd": "VECTOR", "name": "v", "size": -(2**32) + 3, "min": 0.0, "max": 1.0}, "size"),
        ({"cmd": "MIXTURE", "name": "m", "components": -(2**32) + 3}, "components"),
    ],
)
def test_an_int_beyond_the_engines_int_is_refused(stmt, field):
    """The typed methods give the engine an int as a C int, which silently keeps the low 32
    bits (2**32 + 1 becomes 1)."""
    with pytest.raises(ValueError, match=rf"{stmt['cmd']}.*{field}.*2\*\*31"):
        OptQuestSampler(statements=_with_output(X_STMT, stmt), license="")


def test_a_seed_beyond_the_engines_int_is_mapped_and_not_refused():
    sampler = OptQuestSampler(
        statements=_with_output(X_STMT, {"cmd": "RANDOMSEED", "seed": 2**40}), license=""
    )
    assert sampler._model is not None


def test_a_grid_with_far_too_many_steps_gets_the_error_of_a_grid():
    with pytest.raises(ValueError, match=r"2\*\*31 - 1 steps"):
        OptQuestSampler(
            statements=_with_output(
                {"cmd": "DISCRETE", "name": "x", "min": 0.0, "max": 1.0, "step": 1e-28}
            ),
            license="",
        )
    assert _variable_codec.fits_engine_grid(0.0, 1.0, 1e-310) is False


def test_an_engine_error_comes_without_the_java_class_and_the_item_number():
    with pytest.raises(RuntimeError) as caught:
        OptQuestSampler(
            statements=_with_output(
                X_STMT, {"cmd": "STOPIMPROVEMENT", "threshold": 0.5, "cycles": 50}
            ),
            license="",
        )
    message = str(caught.value)
    assert "AUTOSTOPPERCENTIMPROVE" in message
    assert "com.opttek" not in message and ":::" not in message


def _warnings_with_files(study, objective, n_trials):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study.optimize(objective, n_trials=n_trials)
    return [(str(w.message), os.path.basename(w.filename)) for w in caught]


def test_the_warnings_of_the_sampler_point_at_the_sampler():
    """Optuna calls sample_relative() and sample_independent(): a warning one frame up would
    point into Optuna, and a filter on Optuna's warnings would hide it."""
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    caught = _warnings_with_files(study, sphere, 52)
    cap = [f for m, f in caught if "demo-license cap" in m]
    assert cap == ["OptQuestSampler.py"]
    sampler = _sphere_sampler()
    study = optuna.create_study(sampler=sampler)
    caught = _warnings_with_files(
        study, lambda t: t.suggest_float("x", 0.0, 0.001) + t.suggest_float("y", -5, 10), 5
    )
    discarded = [f for m, f in caught if "discarded" in m]
    assert discarded and set(discarded) == {"OptQuestSampler.py"}


def test_a_log_scale_bound_beyond_a_float_is_refused():
    with pytest.raises(ValueError, match=r"'x'.*float"):
        OptQuestSampler(
            search_space={"x": D.IntDistribution(1, 10**400, log=True)},
            directions=[MIN],
            license="",
        )


@pytest.mark.parametrize("value", [None, "abc", [1.0]])
def test_an_output_that_is_not_a_number_is_named(value):
    sampler = OptQuestSampler(statements=_with_output(X_STMT), license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    with pytest.raises(ValueError, match="output 'out'.*number"):
        study.optimize(sampler.wrap(lambda p: {"out": value}), n_trials=1)


@pytest.mark.parametrize(
    "var, element",
    [
        ({"cmd": "BINARYVECTOR", "name": "v", "k": 3}, "v@1"),
        ({"cmd": "BINARYMATRIX", "name": "v", "rows": 2, "cols": 2}, "v@1,1"),
        ({"cmd": "PERMUTATION", "name": "v", "items": ["a", "b", "c", "d"]}, "v@a"),
        ({"cmd": "ITEMSELECTION", "name": "v", "n": 4, "k": 2}, "v@1"),
    ],
)
def test_an_excluded_range_on_an_element_that_the_engine_fixes_is_named(var, element):
    """The engine takes the range and refuses the model only when it initializes it, with an
    error that names no statement."""
    stmt = {"cmd": "EXCLUSIVE", "name": element, "min": 2.0, "max": 3.0}
    with pytest.raises(ValueError, match=rf"EXCLUSIVE '{element}'.*{var['cmd']}"):
        OptQuestSampler(statements=_with_output(var, stmt), license="")


STRADDLING_GRIDS = [(-2e-6, 1 / 3, 9), (-0.999999, 0.25, 4), (-2.999998, 1 / 3, 9)]


@pytest.mark.parametrize("low, step, n", STRADDLING_GRIDS)
@pytest.mark.parametrize("mode", ["plain", "statements"])
def test_the_engines_zero_inside_a_grid_that_straddles_zero_is_moved_onto_the_grid(
    mode, low, step, n
):
    """With a min within 1e-5 below 0 (or a max within 1e-5 above 0), the engine's 0.0 is inside
    the bounds and not a value of the grid: the sampler moves it onto the bound."""
    high = low + n * step
    if mode == "plain":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Optuna moves the computed top onto its grid
            dist = D.FloatDistribution(low, high, step=step)
        sampler = OptQuestSampler(search_space={"x": dist}, directions=[MIN], license="", seed=3)
        objective = lambda t: abs(sampler.suggest_all(t)["x"] - 1.0)  # noqa: E731
    else:
        stmt = {"cmd": "DISCRETE", "name": "x", "min": low, "max": high, "step": step}
        sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
        objective = sampler.wrap(lambda p: {"out": abs(p["x"] - 1.0)})
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    caught = _optimize(study, objective, 45)
    assert not [w for w in caught if "discarded" in w], caught


# Grids with a value within 1e-5 of 0 that is neither 0 nor a bound within half a step of 0.
INTERIOR_NEAR_ZERO_GRIDS = [
    (-1.0, 1.0, 0.166667),
    (-0.5000001, 0.5, 0.25),
    (-9e-06, 1.1e-05, 1e-05),
    (-1.000001, 0.5, 0.5),
    (-0.999999, 1.000001, 0.25),
]


@pytest.mark.parametrize("low, high, step", INTERIOR_NEAR_ZERO_GRIDS)
@pytest.mark.parametrize("mode", ["plain", "statements", "vector", "model"])
def test_the_engines_zero_next_to_a_value_inside_a_grid_is_moved_onto_it(mode, low, high, step):
    """The grid value within 1e-5 of 0 need not be a bound: from -1 to 1 with step 0.166667 it
    is 2e-06. The engine's 0.0 is moved onto it in every mode, also where the registered
    distribution is the range of the grid (model=), which holds 0.0."""
    if mode == "plain":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Optuna moves `high` onto its grid
            dist = D.FloatDistribution(low, high, step=step)
        sampler = OptQuestSampler(search_space={"x": dist}, directions=[MIN], license="", seed=3)
    elif mode in ("statements", "vector"):
        stmt = {"cmd": "DISCRETE", "name": "x", "min": low, "max": high, "step": step}
        if mode == "vector":
            stmt = {**stmt, "cmd": "DISCRETEVECTOR", "size": 1}
        sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
    else:
        model = optquest.OptQuestModel()
        model.set_license("")
        model.add_discrete_variable("x", low, high, step)
        model.add_output_variable("out")
        model.add_minimize_objective("obj", "out")
        sampler = OptQuestSampler(model=model, seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    if mode == "plain":
        caught = _optimize(study, lambda t: abs(sampler.suggest_all(t)["x"] - 1.0), 45)
    else:
        caught = _optimize(
            study,
            sampler.wrap(
                lambda p: {"out": abs(float(p["x"][0] if mode == "vector" else p["x"]) - 1.0)}
            ),
            45,
        )
    assert caught == [], caught
    values = [value for trial in study.trials for value in trial.params.values()]
    assert len(values) > 2 and 0.0 not in values
    for value in values:  # each one is a value of the grid, or the max that the engine keeps
        k = (value - low) / step
        assert abs(k - round(k)) < 1e-8 or value == high, value


@pytest.mark.parametrize(
    "high, step, below", [(1.0, 0.333333, 0.999999), (4.000001, 1.0, 4.0), (4.0000001, 1.0, 4.0)]
)
def test_a_built_model_keeps_the_grid_value_next_to_the_max_it_keeps(high, step, below):
    """Next to a `max` that it keeps, the engine also proposes the grid value below it: each of
    the two reaches the objective, and goes back to the engine, as the engine proposed it."""
    model = optquest.OptQuestModel()
    model.set_license("")
    model.add_discrete_variable("d", 0.0, high, step)
    model.add_output_variable("out")
    model.add_maximize_objective("obj", "out")
    sampler = OptQuestSampler(model=model, seed=4)
    study, caught, fallbacks = _run_counting_fallbacks(sampler, lambda p: {"out": p["d"]}, 30)
    assert caught == [] and fallbacks == []
    values = [t.params["d"] for t in study.trials if t.params]
    assert {below, high} <= set(values) and len(values) == len(set(values))
    recorded = {sol.get_value("d") for sol in model.get_all_solutions()}
    assert {below, high} <= recorded


def test_only_the_engines_zero_is_moved_onto_a_grid():
    fit_class = importlib.import_module(f"{_package}.OptQuestSampler")._FitToDistribution
    # A grid value one step of 1e-6 from the min is a value, not the engine's 0.0.
    fit = fit_class(D.FloatDistribution(0.0, 1e-5), lambda raw: raw, 1e-6)
    assert fit(1e-6) == 1e-6 and fit(9e-6) == 9e-6 and fit(0.0) == 0.0
    # The max that the engine keeps, and the grid value below it.
    fit = fit_class(D.FloatDistribution(0.0, 1.0), lambda raw: raw, 0.333333)
    assert fit(0.999999) == 0.999999 and fit(1.0) == 1.0 and fit(0.0) == 0.0
    # 0.0 goes onto the value next to it: a value inside the grid, or the max that is kept.
    fit = fit_class(D.FloatDistribution(-1.0, 0.833337, step=0.166667), lambda raw: raw, 0.166667)
    assert fit(0.0) == -1.0 + 6 * 0.166667 and fit(-1.0) == -1.0 and fit(0.833337) == 0.833337
    fit = fit_class(D.FloatDistribution(-1.0000005, 1e-6), lambda raw: raw, 0.25)
    assert fit(0.0) == -1.0000005 + 4 * 0.25
    fit = fit_class(D.FloatDistribution(-1.000009, 1e-6), lambda raw: raw, 0.25)
    assert fit(0.0) == 1e-6
    # With no value within 1e-5 of it, 0.0 stays, and Optuna's warning reports it.
    fit = fit_class(D.FloatDistribution(0.5, 2.5, step=0.5), lambda raw: raw, 0.5)
    assert fit(0.0) == 0.0


def test_an_output_beyond_a_float_is_named():
    sampler = OptQuestSampler(statements=_with_output(X_STMT), license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    with pytest.raises(ValueError, match="output 'out'.*number"):
        study.optimize(sampler.wrap(lambda p: {"out": 10**400}), n_trials=1)


def _typed_graph(cmd, predecessors):
    """The describe() line of the variable that the typed method builds."""
    model = optquest.OptQuestModel()
    model.set_license("")
    if cmd == "DAG":
        model.add_dag_variable("g", ["a", "b"], predecessors=predecessors)
    elif cmd == "TREE":
        model.add_tree_variable("g", ["a", "b"], predecessors=predecessors)
    else:
        model.add_topological_sequence_variable("g", ["a", "b"], predecessors)
    return model.describe().splitlines()[0].strip()


@pytest.mark.parametrize("cmd", ["DAG", "TREE", "TOPOLOGICALSEQUENCE"])
@pytest.mark.parametrize(
    "predecessors",
    [{"b": ""}, {"b": ["a", ""]}, {"b": [None]}, {"b": ["a", None]}, {"b": None}, {"b": []}],
)
def test_statements_with_an_empty_predecessor_build_the_typed_methods_model(cmd, predecessors):
    field = "tasks" if cmd == "TOPOLOGICALSEQUENCE" else "nodes"
    stmt = {"cmd": cmd, "name": "g", field: ["a", "b"], "predecessors": predecessors}
    sampler = OptQuestSampler(statements=_with_output(stmt), license="", seed=3)
    built = [line.strip() for line in sampler._model.describe().splitlines()]
    assert _typed_graph(cmd, predecessors) in built


@pytest.mark.parametrize(
    "once",
    [list, set, lambda names: (n for n in names), iter, lambda names: map(str, names)],
    ids=["list", "set", "generator", "iterator", "map"],
)
def test_names_given_as_an_iterator_reach_the_model(once):
    """The statements are read more than once (emit_model() checks them before the model is
    built), and an iterator is empty for the second reader: the precedence and the required
    node were dropped from the model with no error."""
    statements = _with_output(
        {
            "cmd": "TOPOLOGICALSEQUENCE",
            "name": "t",
            "tasks": ["a", "b"],
            "predecessors": {"b": once(["a"])},
        },
        {
            "cmd": "PATH",
            "name": "p",
            "nodes": ["a", "b", "c"],
            "path": {"start": "a", "required": once(["b"])},
        },
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    built = [line.strip() for line in sampler._model.describe().splitlines()]
    assert "TOPOLOGICALSEQUENCE t a,b b,a" in built
    assert "PATH p a,b,c a;;b" in built


@pytest.mark.parametrize(
    "stmt, message",
    [
        ({"cmd": "EXCLUSIVE", "name": "i", "min": "5", "max": "2"}, "`min` .* greater than `max`"),
        ({"cmd": "OBJECTIVEGOAL", "name": "obj", "min": "5", "max": "1"}, "greater than `max`"),
        (
            {"cmd": "OBJECTIVECONFIDENCE", "name": "obj", "ctype": 1, "clevel": 3, "epct": "5"},
            "epct: the error is a fraction",
        ),
    ],
)
def test_a_number_given_as_text_gets_the_checks_of_a_number(stmt, message):
    """emit_model() checks the order and the range of numbers only when they are numbers: with
    "5" and "2", the engine took the range from 5 to 2 and excluded nothing."""
    statements = _with_output({"cmd": "INTEGER", "name": "i", "min": 0, "max": 9})
    as_numbers = {k: float(v) if k in ("min", "max", "epct") else v for k, v in stmt.items()}
    for given in (as_numbers, stmt):
        with pytest.raises(optquest.EmitError, match=message):
            OptQuestSampler(statements=[*statements, given], license="")


def test_numbers_given_as_text_in_the_right_order_are_taken():
    statements = _with_output(
        {"cmd": "INTEGER", "name": "i", "min": "0", "max": "9"},
        {"cmd": "EXCLUSIVE", "name": "i", "min": "2", "max": "5"},
    ) + [{"cmd": "OBJECTIVEGOAL", "name": "obj", "min": "1", "max": "5"}]
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    study, caught, fallbacks = _run_counting_fallbacks(sampler, lambda p: {"out": p["i"]}, 12)
    assert caught == [] and fallbacks == []
    assert {t.params["i"] for t in study.trials if t.params} == {0, 1, 2, 5, 6, 7, 8, 9}


@pytest.mark.parametrize(
    "value, step",
    [(2.0**50, 1.0), (2.0**50, 0.5), (-(2.0**50), 1.0), (1e15, 1.0), (1e15, 0.25), (5.0, 1.0)],
)
def test_a_one_value_grid_far_from_zero_stays_one_value(value, step):
    """Far from zero, a few ulps are as large as the step: a grid of one value was given to the
    engine as a grid of two (and at 2**50, with a max beyond the bound that the engine takes)."""
    line = f"DISCRETE g {value:.0f} {value:.0f} {step:g}"
    plain = OptQuestSampler(
        search_space={"g": D.FloatDistribution(value, value, step=step), **SPHERE_SPACE},
        directions=[MIN],
        license="",
        seed=3,
    )
    stated = OptQuestSampler(
        statements=_with_output(
            {"cmd": "DISCRETE", "name": "g", "min": value, "max": value, "step": step}, X_STMT
        ),
        license="",
        seed=3,
    )
    for sampler in (plain, stated):
        assert line in [text.strip() for text in sampler._model.describe().splitlines()]
        assert (sampler._space["g"].low, sampler._space["g"].high) == (value, value)


def test_spans_a_step_allows_a_rounding_error_and_not_half_a_step():
    spans = _variable_codec.spans_a_step
    assert spans(0.1, 0.3, 0.2) and spans(1e15, 1e15 + 1.0, 1.0) and spans(0.0, 1.0, 1.0)
    assert not spans(1e15, 1e15, 1.0) and not spans(2.0**50, 2.0**50, 0.5)
    assert not spans(1e15, 1e15 + 0.25, 1.0) and not spans(0.0, 0.3, 0.5)


def test_a_numpy_long_double_is_a_python_number():
    """numpy.longdouble.tolist() gives the long double back, where every other numpy number
    gives a Python one."""
    np = pytest.importorskip("numpy")
    plain = _variable_codec.plain_value
    assert type(plain(np.longdouble(0.5))) is float and plain(np.longdouble(0.5)) == 0.5
    assert type(plain(np.clongdouble(0.5))) is complex
    values = plain(np.array([0.0, 1.5], dtype=np.longdouble))
    assert values == [0.0, 1.5] and all(type(v) is float for v in values)
    statements = _with_output(
        {"cmd": "CONTINUOUS", "name": "x", "min": np.longdouble(0), "max": np.longdouble(1)},
        {"cmd": "ENUMERATION", "name": "en", "values": np.array([0.0, 1.5], dtype=np.longdouble)},
    )
    sampler = OptQuestSampler(statements=statements, license="", seed=3)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    assert _optimize(study, sampler.wrap(lambda p: {"out": p["x"] + p["en"]}), 3) == []
    assert all(type(t.params["x"]) is float for t in study.trials)


# ---------------------------------------------------------------------------------------------
# EXCLUSIVE ranges far from zero
# ---------------------------------------------------------------------------------------------

# Ten values, base + 0 to base + 9, where 1e-9 of the base is more than the distance between
# two of them (10 at 1e10), and near 2**50, where a few ulps are.
FAR_BASES = [2.0**31 - 5, 1e10, 2.0**50 - 16]
FAR_KINDS = ["INTEGER", "DISCRETE", "ENUMERATION", "CONTINUOUS"]


def _ten_values(kind, base):
    if kind == "INTEGER":
        return {"cmd": "INTEGER", "name": "v", "min": int(base), "max": int(base) + 9}
    if kind == "DISCRETE":
        return {"cmd": "DISCRETE", "name": "v", "min": base, "max": base + 9, "step": 1.0}
    if kind == "ENUMERATION":
        return {"cmd": "ENUMERATION", "name": "v", "values": [base + k for k in range(10)]}
    return {"cmd": "CONTINUOUS", "name": "v", "min": base, "max": base + 9}


def _far_samplers(kind, base, ranges):
    """The sampler for ten values and the excluded `ranges`, from statements= and from model=,
    or the ValueError that refused it."""
    exclusive = [{"cmd": "EXCLUSIVE", "name": "v", "min": lo, "max": hi} for lo, hi in ranges]

    def stated():
        statements = _with_output(_ten_values(kind, base), *exclusive, X_STMT)
        return OptQuestSampler(statements=statements, license="", seed=3)

    def built():
        model = optquest.OptQuestModel()
        model.set_license("")
        if kind == "INTEGER":
            model.add_integer_variable("v", int(base), int(base) + 9)
        elif kind == "DISCRETE":
            model.add_discrete_variable("v", base, base + 9, 1.0)
        elif kind == "ENUMERATION":
            model.add_enumeration_variable("v", [base + k for k in range(10)])
        else:
            model.add_continuous_variable("v", base, base + 9)
        for lo, hi in ranges:
            model.add_variable_exclusive_range("v", lo, hi)
        model.add_continuous_variable("x", -1.0, 1.0)
        model.add_output_variable("out")
        model.add_minimize_objective("obj", "out")
        return OptQuestSampler(model=model, seed=3)

    made = []
    for make in (stated, built):
        try:
            made.append(make())
        except ValueError as exc:
            made.append(exc)
    return made


def _far_values(sampler, base):
    """The values of `v` that the objective gets, as offsets from the base."""
    seen = []

    def sim(p):
        seen.append(p["v"] - base)
        return {"out": p["x"]}

    _study, caught, fallbacks = _run_counting_fallbacks(sampler, sim, 30)
    assert caught == [] and fallbacks == []
    return seen


def _exclusive_edges(sampler, base):
    lines = [text.split() for text in sampler._model.describe().splitlines()]
    return [(float(t[2]) - base, float(t[3]) - base) for t in lines if t[:1] == ["EXCLUSIVE"]]


@pytest.mark.parametrize("base", FAR_BASES)
@pytest.mark.parametrize("kind", FAR_KINDS)
def test_an_exclusive_range_far_from_zero_keeps_its_values_out(kind, base):
    """An edge between two values, or beyond the last one, counted as a value when the rounding
    error of the bounds (1e-9 of them) was more than the distance between two values: the range
    was built as it was given, and the objective got a value inside it."""
    edges = [(3, 6), (3.5, 5.5), (2.75, 6.25), (3, 9.25), (3, 10), (5, 9.5), (-0.25, 4), (-1, 4)]
    for lower, upper in edges:
        low, high = base + lower, base + upper
        for sampler in _far_samplers(kind, base, [(low, high)]):
            if isinstance(sampler, ValueError):
                continue
            inside = [v for v in _far_values(sampler, base) if lower < v < upper]
            assert inside == [], (lower, upper, inside)
    # An edge beyond the last value, with no value to move to, is refused in both modes.
    for lower, upper in [(3, 9.25), (3, 10), (-1, 4)]:
        refused = _far_samplers(kind, base, [(base + lower, base + upper)])
        assert all(isinstance(r, ValueError) and "with no value beyond" in str(r) for r in refused)
    if kind == "CONTINUOUS":
        return
    # An edge between two values moves outward (statements=) or is refused (model=).
    stated, built = _far_samplers(kind, base, [(base + 3.5, base + 5.5)])
    assert _exclusive_edges(stated, base) == [(3.0, 6.0)]
    assert set(_far_values(stated, base)) == {0, 1, 2, 3, 6, 7, 8, 9}
    assert isinstance(built, ValueError) and "Declare the range from" in str(built)
    for sampler in _far_samplers(kind, base, [(base + 3, base + 6)]):
        assert _exclusive_edges(sampler, base) == [(3.0, 6.0)]
        assert set(_far_values(sampler, base)) == {0, 1, 2, 3, 6, 7, 8, 9}


@pytest.mark.parametrize("base", FAR_BASES)
@pytest.mark.parametrize("kind", FAR_KINDS)
def test_overlapping_exclusive_ranges_far_from_zero_are_one_range(kind, base):
    """Two ranges that overlapped by less than the rounding error of the bounds were taken for
    ranges that touch, and the value at the shared edge reached the objective."""
    for first, second in [((1, 5), (3, 8)), ((1, 5), (4, 8)), ((1.5, 4.5), (4, 7.5))]:
        ranges = [(base + first[0], base + first[1]), (base + second[0], base + second[1])]
        stated, built = _far_samplers(kind, base, ranges)
        # A built model is refused: for the overlap, or first for an edge between two values.
        whole = kind == "CONTINUOUS" or first[0] == int(first[0])
        assert isinstance(built, ValueError) and ("overlap" in str(built)) == whole
        if kind != "CONTINUOUS":
            assert _exclusive_edges(stated, base) == [(1.0, 8.0)]
        inside = [v for v in _far_values(stated, base) if first[0] < v < second[1]]
        assert inside == []
    # Ranges that touch, and ranges apart, are kept as they are.
    for pair in ([(1, 5), (5, 8)], [(1, 4), (6, 8)]):
        ranges = [(base + lo, base + hi) for lo, hi in pair]
        for sampler in _far_samplers(kind, base, ranges):
            assert _exclusive_edges(sampler, base) == [(float(lo), float(hi)) for lo, hi in pair]
            values = _far_values(sampler, base)
            assert not [v for v in values if any(lo < v < hi for lo, hi in pair)]
