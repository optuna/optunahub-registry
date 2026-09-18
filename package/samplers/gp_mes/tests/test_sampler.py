"""Tests for the max-value entropy search sampler.

Uses ``optuna.testing.pytest_samplers`` for the generic sampler contract, plus tests for
behaviour specific to this package: the two max-value samplers, argument validation, the
multi-objective and constrained fallbacks, and reproducibility.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.testing.pytest_samplers import BasicSamplerTestCase
from optuna.testing.pytest_samplers import MultiObjectiveSamplerTestCase
from optuna.testing.pytest_samplers import RelativeSamplerTestCase
import optunahub
import pytest
import torch


_mod = optunahub.load_local_module(package="samplers/gp_mes", registry_root="package/")
MESSampler = _mod.MESSampler
GIBBONSampler = _mod.GIBBONSampler
_MaxValueEntropySearch = _mod.sampler._MaxValueEntropySearch
_GIBBON = _mod.sampler._GIBBON


# The generic test cases run with very few completed trials, so ``n_startup_trials=1``
# forces the GP code path rather than random startup sampling.


class TestMESSamplerGumbel(
    BasicSamplerTestCase, RelativeSamplerTestCase, MultiObjectiveSamplerTestCase
):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: MESSampler(max_value_sampler="gumbel", n_startup_trials=1, seed=42)


class TestMESSamplerPosterior(
    BasicSamplerTestCase, RelativeSamplerTestCase, MultiObjectiveSamplerTestCase
):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: MESSampler(
            max_value_sampler="posterior", n_startup_trials=1, seed=42, n_representer_points=64
        )


class TestGIBBONSampler(
    BasicSamplerTestCase, RelativeSamplerTestCase, MultiObjectiveSamplerTestCase
):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: GIBBONSampler(n_startup_trials=1, seed=42, n_representer_points=64)


# --------------------------------------------------------------------------- #
# Package-specific tests
# --------------------------------------------------------------------------- #


def _sphere(trial: optuna.Trial) -> float:
    return sum(trial.suggest_float(f"x{i}", -5, 5) ** 2 for i in range(3))


def _bi_objective(trial: optuna.Trial) -> tuple[float, float]:
    x = [trial.suggest_float(f"x{i}", -5, 5) for i in range(3)]
    return sum(v**2 for v in x), -sum(v**2 for v in x)


@pytest.mark.parametrize("max_value_sampler", ["gumbel", "posterior"])
def test_optimization_runs(max_value_sampler: str) -> None:
    study = optuna.create_study(
        sampler=MESSampler(max_value_sampler=max_value_sampler, seed=42, n_representer_points=64)
    )
    study.optimize(_sphere, n_trials=20)
    assert len(study.trials) == 20
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_value_sampler": "not-a-sampler"},
        {"n_max_value_samples": 0},
        {"n_representer_points": 0},
    ],
)
def test_invalid_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MESSampler(**kwargs)


def test_reproducibility() -> None:
    results: list[list[float]] = []
    for _ in range(2):
        study = optuna.create_study(sampler=MESSampler(seed=42))
        study.optimize(_sphere, n_trials=15)
        results.append([t.value for t in study.trials])

    np.testing.assert_array_equal(results[0], results[1])


def test_multi_objective_fallback() -> None:
    """Multi-objective studies fall back to the parent GPSampler."""
    study = optuna.create_study(
        directions=["minimize", "maximize"], sampler=MESSampler(n_startup_trials=1, seed=42)
    )
    study.optimize(_bi_objective, n_trials=8)

    assert len(study.trials) == 8
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_constraints_func_fallback() -> None:
    """A constrained study is accepted and falls back to the parent GPSampler."""
    study = optuna.create_study(sampler=MESSampler(seed=42, constraints_func=lambda t: [0.0]))
    study.optimize(_sphere, n_trials=15)

    assert len(study.trials) == 15


def test_categorical_search_space() -> None:
    """Categorical dimensions carry raw indices in the normalized space."""

    def objective(trial: optuna.Trial) -> float:
        c = trial.suggest_categorical("c", ["a", "b", "c"])
        x = trial.suggest_float("x", -5, 5)
        return x**2 + {"a": 0.0, "b": 1.0, "c": 2.0}[c]

    study = optuna.create_study(sampler=MESSampler(seed=42, n_startup_trials=3))
    study.optimize(objective, n_trials=15)
    assert len(study.trials) == 15


def test_acqf_equals_mutual_information() -> None:
    """The acquisition matches the closed-form mutual information I(f(x); y*).

    Checked against an independent evaluation via ``math.erfc`` rather than the tensor
    expression used by the implementation.
    """
    from optuna._gp import gp
    from optuna._gp import prior
    from optuna._gp import search_space as gp_search_space
    from optuna.distributions import FloatDistribution

    rng = np.random.default_rng(0)
    X = rng.random((12, 2))
    Y = np.sin(6 * X[:, 0])
    Y = (Y - Y.mean()) / Y.std()
    gpr = gp.fit_kernel_params(
        X=X,
        Y=Y,
        is_categorical=np.zeros(2, dtype=bool),
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=False,
    )
    space = gp_search_space.SearchSpace(
        {"x0": FloatDistribution(0, 1), "x1": FloatDistribution(0, 1)}
    )
    y_star = torch.tensor([2.0], dtype=torch.float64)
    acqf = _MaxValueEntropySearch(gpr, space, y_star)

    x = torch.from_numpy(rng.random((6, 2)))
    got = acqf.eval_acqf(x).detach().numpy()

    mean, var = gpr.posterior(x)
    mean_np = mean.detach().numpy()
    sigma_np = np.sqrt(var.detach().numpy() + 1e-12)
    gamma = (2.0 - mean_np) / sigma_np
    cdf = np.array([0.5 * math.erfc(-g / math.sqrt(2.0)) for g in gamma])
    pdf = np.exp(-0.5 * gamma**2) / math.sqrt(2.0 * math.pi)
    expected = gamma * pdf / (2.0 * cdf) - np.log(cdf)

    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_acqf_is_non_negative_and_batch_separable() -> None:
    """Mutual information is non-negative, and rows of a batch must not interact."""
    from optuna._gp import gp
    from optuna._gp import prior
    from optuna._gp import search_space as gp_search_space
    from optuna.distributions import FloatDistribution

    rng = np.random.default_rng(1)
    X = rng.random((10, 2))
    Y = (X[:, 0] - 0.5) ** 2
    Y = (Y - Y.mean()) / Y.std()
    gpr = gp.fit_kernel_params(
        X=X,
        Y=Y,
        is_categorical=np.zeros(2, dtype=bool),
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=False,
    )
    space = gp_search_space.SearchSpace(
        {"x0": FloatDistribution(0, 1), "x1": FloatDistribution(0, 1)}
    )
    acqf = _MaxValueEntropySearch(gpr, space, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))

    batch = torch.from_numpy(rng.random((8, 2)))
    values = acqf.eval_acqf(batch)
    assert bool(torch.all(values >= -1e-9))
    assert bool(torch.isfinite(values).all())

    perturbed = batch.clone()
    perturbed[2] = torch.from_numpy(rng.random(2))
    other = acqf.eval_acqf(perturbed)
    kept = [i for i in range(8) if i != 2]
    np.testing.assert_allclose(
        values[kept].detach().numpy(), other[kept].detach().numpy(), rtol=0, atol=1e-15
    )


def test_max_values_are_above_incumbent() -> None:
    """Sampled maxima must not fall below the best observation."""
    from optuna._gp import gp
    from optuna._gp import prior
    from optuna._gp import search_space as gp_search_space
    from optuna.distributions import FloatDistribution

    rng = np.random.default_rng(2)
    X = rng.random((10, 2))
    Y = X[:, 0]
    Y = (Y - Y.mean()) / Y.std()
    gpr = gp.fit_kernel_params(
        X=X,
        Y=Y,
        is_categorical=np.zeros(2, dtype=bool),
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=False,
    )
    space = gp_search_space.SearchSpace(
        {"x0": FloatDistribution(0, 1), "x1": FloatDistribution(0, 1)}
    )
    y_best = float(Y.max())

    for sample in (
        _mod.sampler._sample_max_values_by_gumbel,
        _mod.sampler._sample_max_values_by_posterior,
    ):
        values = sample(gpr, space, 64, 128, np.random.RandomState(0), y_best).numpy()
        assert values.shape == (64,)
        assert np.all(values >= y_best)
        assert np.all(np.isfinite(values))


# --------------------------------------------------------------------------- #
# GIBBON
# --------------------------------------------------------------------------- #


def _fitted_gp(n: int = 12, dim: int = 2, seed: int = 0):  # type: ignore[no-untyped-def]
    from optuna._gp import gp
    from optuna._gp import prior
    from optuna._gp import search_space as gp_search_space
    from optuna.distributions import FloatDistribution

    rng = np.random.default_rng(seed)
    X = rng.random((n, dim))
    Y = np.sin(5 * X[:, 0]) + X[:, 1] ** 2
    Y = (Y - Y.mean()) / Y.std()
    gpr = gp.fit_kernel_params(
        X=X,
        Y=Y,
        is_categorical=np.zeros(dim, dtype=bool),
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=False,
    )
    space = gp_search_space.SearchSpace({f"x{i}": FloatDistribution(0, 1) for i in range(dim)})
    return gpr, space


def test_gibbon_is_a_lower_bound_on_exact_mes() -> None:
    """Theorem 3 of Moss et al. bounds the information gain from below.

    With no pending trials the batch is a single point, so GIBBON must sit at or below the
    exact value that ``_MaxValueEntropySearch`` computes.
    """
    gpr, space = _fitted_gp()
    y_star = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    x = torch.from_numpy(np.random.default_rng(1).random((40, 2)))

    exact = _MaxValueEntropySearch(gpr, space, y_star).eval_acqf(x).detach().numpy()
    bound = _GIBBON(gpr, space, y_star, pending=None).eval_acqf(x).detach().numpy()

    assert np.all(bound <= exact + 1e-9)
    assert np.all(np.isfinite(bound))


def test_gibbon_penalises_candidates_near_pending_trials() -> None:
    """The correlation term is the point of the acquisition: it must repel."""
    gpr, space = _fitted_gp()
    y_star = torch.tensor([2.0], dtype=torch.float64)
    pending = np.array([[0.5, 0.5]])

    acqf = _GIBBON(gpr, space, y_star, pending=pending)
    near = torch.tensor([[0.5, 0.5]], dtype=torch.float64)  # duplicate of the pending point
    far = torch.tensor([[0.05, 0.95]], dtype=torch.float64)

    assert float(acqf.eval_acqf(near)) < float(acqf.eval_acqf(far))

    # Without the pending point the ordering is driven by information alone, so the
    # duplicate is no longer disfavoured for being a duplicate.
    plain = _GIBBON(gpr, space, y_star, pending=None)
    penalty_near = float(plain.eval_acqf(near)) - float(acqf.eval_acqf(near))
    penalty_far = float(plain.eval_acqf(far)) - float(acqf.eval_acqf(far))
    assert penalty_near > penalty_far


def test_gibbon_matches_a_direct_joint_posterior() -> None:
    """The bordered-determinant shortcut must equal a full joint posterior per candidate.

    The reference is built from scratch: eq. (7) of Moss et al. with the observation
    covariance carrying the fitted noise, gamma formed from the latent posterior, and
    rho^2 the ratio between them.
    """
    gpr, space = _fitted_gp()
    y_star = torch.tensor([1.5], dtype=torch.float64)
    rng = np.random.default_rng(3)
    pending = rng.random((3, 2))
    cand = torch.from_numpy(rng.random((5, 2)))
    noise = float(gpr.noise_var)

    acqf = _GIBBON(gpr, space, y_star, pending=pending)
    got = acqf.eval_acqf(cand).detach().numpy()

    pending_t = torch.from_numpy(pending)
    expected = []
    for b in range(cand.shape[0]):
        full = torch.cat([pending_t, cand[b : b + 1]], dim=0)
        mean, cov_latent = gpr.posterior(full, joint=True)
        cov_obs = cov_latent + noise * torch.eye(full.shape[0], dtype=cov_latent.dtype)
        sd_obs = torch.sqrt(torch.diagonal(cov_obs))
        corr = cov_obs / torch.outer(sd_obs, sd_obs)
        latent_var = torch.diagonal(cov_latent)
        rho_sq = latent_var / torch.diagonal(cov_obs)
        gamma = (y_star[0] - mean) / torch.sqrt(latent_var)
        log_cdf = torch.special.log_ndtr(gamma)
        ratio = torch.exp(-0.5 * gamma**2 - 0.5 * math.log(2 * math.pi) - log_cdf)
        inner = torch.clamp(1.0 - rho_sq * ratio * (gamma + ratio), min=1e-300)
        ig = 0.5 * torch.linalg.slogdet(corr)[1] - 0.5 * torch.log(inner).sum()
        expected.append(float(ig))

    np.testing.assert_allclose(got, np.array(expected), rtol=1e-6, atol=1e-9)


def test_gibbon_is_batch_separable() -> None:
    """Rows of the optimizer's batch must not interact, even with pending trials."""
    gpr, space = _fitted_gp()
    acqf = _GIBBON(
        gpr,
        space,
        torch.tensor([1.0, 2.0], dtype=torch.float64),
        pending=np.array([[0.3, 0.7], [0.8, 0.2]]),
    )
    rng = np.random.default_rng(4)
    batch = torch.from_numpy(rng.random((8, 2)))
    values = acqf.eval_acqf(batch)

    perturbed = batch.clone()
    perturbed[3] = torch.from_numpy(rng.random(2))
    other = acqf.eval_acqf(perturbed)
    kept = [i for i in range(8) if i != 3]
    np.testing.assert_allclose(
        values[kept].detach().numpy(), other[kept].detach().numpy(), rtol=0, atol=1e-12
    )


def test_gibbon_gradients_flow() -> None:
    """optim_mixed differentiates eval_acqf(x).sum(); the pending path must not break it."""
    gpr, space = _fitted_gp()
    acqf = _GIBBON(
        gpr,
        space,
        torch.tensor([2.0], dtype=torch.float64),
        pending=np.array([[0.4, 0.6]]),
    )
    x = torch.from_numpy(np.random.default_rng(5).random((6, 2))).requires_grad_(True)
    acqf.eval_acqf(x).sum().backward()
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())


def test_gibbon_runs_with_concurrent_trials() -> None:
    """The batch path is exercised only when trials are left running."""
    sampler = GIBBONSampler(seed=42, n_startup_trials=4, n_representer_points=64)
    study = optuna.create_study(sampler=sampler)
    for _ in range(4):
        trial = study.ask({"x0": optuna.distributions.FloatDistribution(-5, 5)})
        study.tell(trial, float(trial.params["x0"] ** 2))

    pending = [study.ask({"x0": optuna.distributions.FloatDistribution(-5, 5)}) for _ in range(3)]
    assert len(pending) == 3
    nxt = study.ask({"x0": optuna.distributions.FloatDistribution(-5, 5)})
    assert "x0" in nxt.params


def test_gibbon_handles_coincident_pending_points() -> None:
    """Regression: duplicate pending trials left the pending covariance singular.

    Categorical search spaces make coincident pending points common, since two running
    trials can map to the same normalized point.
    """
    gpr, space = _fitted_gp()
    y_star = torch.tensor([2.0], dtype=torch.float64)
    duplicated = np.array([[0.5, 0.5], [0.5, 0.5], [0.2, 0.8]])

    acqf = _GIBBON(gpr, space, y_star, pending=duplicated)
    values = acqf.eval_acqf(torch.from_numpy(np.random.default_rng(6).random((5, 2))))
    assert bool(torch.isfinite(values).all())

    # Redundancy among the pending set should depress the score relative to a
    # non-redundant set of the same size.
    distinct = np.array([[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
    spread = _GIBBON(gpr, space, y_star, pending=distinct)
    probe = torch.tensor([[0.4, 0.6]], dtype=torch.float64)
    assert float(acqf.eval_acqf(probe)) < float(spread.eval_acqf(probe))
