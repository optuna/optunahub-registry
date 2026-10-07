"""Tests for CatCmawmSampler.

Covers the mapping from ``cmaes.CatCMAwM``'s internal solution representation back to
Optuna's parameter dictionary. A reversed mapping is silently tolerated by Optuna --
``sample_relative()`` may legally return a partial dict -- so these tests assert on the
keys of the returned dict and on whether ``sample_independent()`` is used, rather than
relying on an exception being raised.
"""

from __future__ import annotations

import math
from typing import Any

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import RandomSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
import optunahub
import pytest


_module = optunahub.load_local_module(package="samplers/catcmawm", registry_root="package/")
CatCmawmSampler = _module.CatCmawmSampler
# A second ``load_local_module`` call leaves the submodule unbound, so reuse this object.
_catcmawm = _module.catcmawm


SEARCH_SPACE: dict[str, BaseDistribution] = {
    "x1": FloatDistribution(-5, 5),
    "x2": FloatDistribution(-5, 5),
    "z1": IntDistribution(-1, 1),
    "z2": IntDistribution(-2, 2),
    "c1": CategoricalDistribution([0, 1, 2]),
    "c2": CategoricalDistribution([0, 1, 2]),
}


class _TrackingRandomSampler(RandomSampler):
    """A ``RandomSampler`` that records which parameters it was asked to sample."""

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.calls: list[tuple[int, str]] = []

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        self.calls.append((trial.number, param_name))
        return super().sample_independent(study, trial, param_name, param_distribution)


def _objective(trial: optuna.Trial) -> float:
    value = 0.0
    for name, distribution in SEARCH_SPACE.items():
        if isinstance(distribution, FloatDistribution):
            value += trial.suggest_float(name, distribution.low, distribution.high) ** 2
        elif isinstance(distribution, IntDistribution):
            value += trial.suggest_int(name, distribution.low, distribution.high) ** 2
        else:
            assert isinstance(distribution, CategoricalDistribution)
            choice = trial.suggest_categorical(name, distribution.choices)
            assert isinstance(choice, int)
            value += choice
    return value


def test_sample_relative_returns_all_param_names() -> None:
    # The returned dict must be keyed by parameter name, for every parameter type.
    sampler = CatCmawmSampler(search_space=SEARCH_SPACE, seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=1)

    trial = study.ask(SEARCH_SPACE)
    params = sampler.sample_relative(study, trial, SEARCH_SPACE)

    assert set(params.keys()) == set(SEARCH_SPACE.keys())


def test_relative_sampling_is_used_for_all_param_types() -> None:
    # Once the search space is inferred, no parameter should fall back to the
    # independent sampler.
    independent_sampler = _TrackingRandomSampler(seed=0)
    sampler = CatCmawmSampler(seed=0, independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=3)

    fallbacks_after_first_trial = [c for c in independent_sampler.calls if c[0] > 0]
    assert fallbacks_after_first_trial == []


def test_suggested_params_match_proposed_solution() -> None:
    # The optimizer is updated via ``tell()`` using the vectors stored in ``user_attrs``,
    # so those vectors must correspond to the values actually evaluated.
    sampler = CatCmawmSampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=5)

    float_names = [n for n, d in SEARCH_SPACE.items() if isinstance(d, FloatDistribution)]
    int_names = [n for n, d in SEARCH_SPACE.items() if isinstance(d, IntDistribution)]

    for trial in study.trials:
        if "x" not in trial.user_attrs:  # First trial, before the search space is known.
            continue
        for proposed, name in zip(trial.user_attrs["x"], sorted(float_names)):
            assert proposed == pytest.approx(trial.params[name])
        for proposed, name in zip(trial.user_attrs["z"], sorted(int_names)):
            assert proposed == trial.params[name]


@pytest.mark.parametrize("distribution_types", ["f", "i", "c", "fi", "fc", "ic", "fic"])
def test_mixed_search_spaces(distribution_types: str) -> None:
    # ``solution.x`` and ``solution.z`` are empty when the corresponding parameter type
    # is absent, which is handled by separate guarded branches.
    def objective(trial: optuna.Trial) -> float:
        value = 0.0
        if "f" in distribution_types:
            value += trial.suggest_float("f1", -5, 5) ** 2
            value += trial.suggest_float("f2", -5, 5) ** 2
        if "i" in distribution_types:
            value += trial.suggest_int("i1", -3, 3) ** 2
            value += trial.suggest_int("i2", -3, 3) ** 2
        if "c" in distribution_types:
            value += float(trial.suggest_categorical("c1", [0, 1, 2]))
            value += float(trial.suggest_categorical("c2", [0, 1, 2]))
        return value

    study = optuna.create_study(sampler=CatCmawmSampler(seed=0))
    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


# --- log=True and float step (see #423) -------------------------------------------------


def _log_objective(trial: optuna.Trial) -> float:
    # Optimum at lr=1e-4, wd=1e-2 -- the bottom 0.1% of the range in linear space, so a
    # sampler that ignores ``log`` collapses onto the lower bound.
    lr = trial.suggest_float("lr", 1e-6, 1.0, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1.0, log=True)
    return (math.log10(lr) + 4) ** 2 + (math.log10(wd) + 2) ** 2


def _log_int_objective(trial: optuna.Trial) -> float:
    # Optimum at n=10, m=100.
    n = trial.suggest_int("n", 1, 1000, log=True)
    m = trial.suggest_int("m", 1, 1000, log=True)
    return (math.log10(n) - 1) ** 2 + (math.log10(m) - 2) ** 2


def test_stepped_float_does_not_fall_back() -> None:
    # Without the step grid in ``z_space`` the returned value is off-grid, so Optuna
    # rejects it and falls back on every trial.
    def objective(trial: optuna.Trial) -> float:
        a = trial.suggest_float("a", -5, 5, step=0.5)
        b = trial.suggest_float("b", -5, 5)
        return a**2 + b**2

    independent_sampler = _TrackingRandomSampler(seed=0)
    study = optuna.create_study(
        sampler=CatCmawmSampler(seed=0, independent_sampler=independent_sampler)
    )
    study.optimize(objective, n_trials=30)

    assert [c for c in independent_sampler.calls if c[0] > 0] == []


@pytest.mark.parametrize(
    "objective,threshold",
    [(_log_objective, 0.5), (_log_int_objective, 0.1)],
)
def test_log_parameters_are_optimized_in_log_space(objective: Any, threshold: float) -> None:
    # Averaged over seeds for stability: unpatched scores ~3 here, patched below 0.1.
    best_values = []
    for seed in range(5):
        study = optuna.create_study(sampler=CatCmawmSampler(seed=seed))
        study.optimize(objective, n_trials=40)
        best_values.append(study.best_value)

    assert sum(best_values) / len(best_values) < threshold


@pytest.mark.parametrize(
    "distribution",
    [
        FloatDistribution(1e-6, 1.0, log=True),
        FloatDistribution(-5, 5, step=0.5),
        FloatDistribution(-1, 2, step=0.1),
        IntDistribution(1, 1000, log=True),
        IntDistribution(0, 98, step=7),
    ],
)
def test_suggested_values_are_valid(distribution: BaseDistribution) -> None:
    # Optuna silently replaces a rejected value, so this guards the transforms against
    # returning off-grid or out-of-range results.
    search_space: dict[str, BaseDistribution] = {
        "p": distribution,
        "q": FloatDistribution(-5, 5),
    }
    sampler = CatCmawmSampler(search_space=search_space, seed=0)
    study = optuna.create_study(sampler=sampler)

    for i in range(10):
        trial = study.ask(search_space)
        params = sampler.sample_relative(study, trial, search_space)
        assert set(params.keys()) == set(search_space.keys())
        value = params["p"]
        assert distribution._contains(distribution.to_internal_repr(value))
        # A constant objective would make CatCMAwM's ranking degenerate.
        study.tell(trial, float(i))


def test_grid_of_uses_log_spacing_for_log_int() -> None:
    grid = _catcmawm._grid_of(IntDistribution(1, 1000, log=True))
    assert grid[0] == pytest.approx(math.log(1))
    assert grid[-1] == pytest.approx(math.log(1000))
    assert all(
        _catcmawm._untransform_z(v, IntDistribution(1, 1000, log=True)) == i + 1
        for i, v in enumerate(grid)
    )


def test_grid_of_returns_exact_grid_for_stepped_float() -> None:
    distribution = FloatDistribution(-5, 5, step=0.5)
    grid = _catcmawm._grid_of(distribution)
    assert len(grid) == 21
    assert grid[0] == pytest.approx(-5.0)
    assert grid[-1] == pytest.approx(5.0)
    assert all(distribution._contains(v) for v in grid)


def test_untransform_x_inverts_the_log_transform() -> None:
    distribution = FloatDistribution(1e-6, 1.0, log=True)
    for external in (1e-6, 1e-4, 1e-2, 1.0):
        assert _catcmawm._untransform_x(math.log(external), distribution) == pytest.approx(
            external
        )


@pytest.mark.parametrize(
    "distribution",
    [FloatDistribution(1e-6, 1.0, log=True), IntDistribution(1, 1000, log=True)],
)
def test_untransform_clamps_to_bounds(distribution: BaseDistribution) -> None:
    # ``exp`` can overshoot a bound by a rounding error.
    untransform = (
        _catcmawm._untransform_x
        if isinstance(distribution, FloatDistribution)
        else _catcmawm._untransform_z
    )
    below = untransform(math.log(distribution.low) - 1e-9, distribution)
    above = untransform(math.log(distribution.high) + 1e-9, distribution)
    assert distribution.low <= below <= distribution.high
    assert distribution.low <= above <= distribution.high
