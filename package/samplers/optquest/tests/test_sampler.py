# mypy: ignore-errors
"""OptQuestSampler tests, run against the engine installed by `pip install optquest`.

Every test builds its own sampler (its own engine instance) and asks for fewer than 50 trials,
so the suite runs in demo mode and needs no license. The engine is deterministic for a given
model and seed, so the assertions do not depend on the search path.

Run from the root of the registry: `pytest package/samplers/optquest/tests`
"""

from collections import Counter
import importlib
import math
import warnings

import optquest
import optuna
import optunahub
import pytest


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


def test_seed_is_deterministic_and_a_non_positive_seed_is_refused():
    runs = []
    for _ in range(2):
        study = optuna.create_study(sampler=_sphere_sampler(seed=11))
        study.optimize(sphere, n_trials=6)
        runs.append([(t.params["x"], t.params["y"]) for t in study.trials])
    assert runs[0] == runs[1]
    with pytest.raises(ValueError, match="seed must be a positive int"):
        OptQuestSampler(search_space=SPHERE_SPACE, directions=[MIN], license="", seed=0)


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


def test_manual_objective_refuses_a_model_with_an_unlinked_output():
    sampler = OptQuestSampler(statements=CONSTRAINED, license="", seed=5)
    study = optuna.create_study(sampler=sampler, directions=sampler.directions)
    with pytest.raises(NotImplementedError, match="\\['budget'\\]"):
        study.optimize(
            lambda t: t.suggest_float("x", -5, 10) ** 2 + t.suggest_float("y", -5, 10) ** 2,
            n_trials=1,
        )


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
