"""Max-value entropy search (MES) sampler.

Implements the acquisition function of Wang & Jegelka, "Max-value Entropy Search for
Efficient Bayesian Optimization", ICML 2017, on top of Optuna's built-in ``GPSampler``.
"""

from __future__ import annotations

import math
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna.samplers._gp.sampler import _standardize_values
from optuna.samplers._gp.sampler import GPSampler
from optuna.study import StudyDirection


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import optuna._gp.acqf as acqf_module
    import optuna._gp.gp as gp
    import optuna._gp.search_space as gp_search_space
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    gp = _LazyImport("optuna._gp.gp")
    acqf_module = _LazyImport("optuna._gp.acqf")


__all__ = ["MESSampler"]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)

# Gumbel quantile constants: for F(z) = exp(-exp(-(z - a) / b)),
# z_q = a - b * log(-log(q)).
_GUMBEL_LOWER_Q = 0.25
_GUMBEL_UPPER_Q = 0.75
_GUMBEL_LOWER_COEF = math.log(-math.log(_GUMBEL_LOWER_Q))  # +0.3266
_GUMBEL_UPPER_COEF = math.log(-math.log(_GUMBEL_UPPER_Q))  # -1.2459


class _MaxValueEntropySearch(acqf_module.BaseAcquisitionFunc):
    """Max-value entropy search acquisition function.

    Scores a candidate by the mutual information between its observation and the maximum
    value ``y* = max_x f(x)``. With ``gamma = (y* - mu(x)) / sigma(x)``, and ``phi``, ``Psi``
    the standard normal PDF and CDF, Wang & Jegelka's eq. 6 gives

        alpha(x) = mean over y* of [ gamma * phi(gamma) / (2 * Psi(gamma)) - log Psi(gamma) ]

    The max-value samples ``y*`` are drawn once by the sampler and frozen here, so that
    ``eval_acqf`` is deterministic and depends on each batch row independently. Both
    properties are required by ``optuna._gp.optim_mixed``, which optimizes the acquisition
    with a batched L-BFGS-B that differentiates ``eval_acqf(x).sum()``.
    """

    def __init__(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        max_value_samples: torch.Tensor,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr = gpr
        self._max_value_samples = max_value_samples
        self._stabilizing_noise = stabilizing_noise
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        sigma = torch.sqrt(var + self._stabilizing_noise)

        # Broadcast the (n_samples,) max-values against mean of shape x.shape[:-1].
        y_star = self._max_value_samples.reshape(
            (-1,) + (1,) * mean.ndim
        )  # (n_samples, *mean.shape)
        gamma = (y_star - mean) / sigma

        log_cdf = torch.special.log_ndtr(gamma)
        log_pdf = -0.5 * gamma**2 - _LOG_SQRT_2PI
        # exp(log_pdf - log_cdf) is the inverse Mills ratio phi/Psi. Forming it in log space
        # keeps the far-negative-gamma tail finite, where phi and Psi both underflow to zero.
        inverse_mills_ratio = torch.exp(log_pdf - log_cdf)

        acqf_values = 0.5 * gamma * inverse_mills_ratio - log_cdf
        return torch.mean(acqf_values, dim=0)


def _sample_max_values_by_gumbel(
    gpr: gp.GPRegressor,
    search_space: gp_search_space.SearchSpace,
    n_samples: int,
    n_grid_points: int,
    rng: np.random.RandomState,
    y_best: float,
) -> torch.Tensor:
    """Sample maxima using the Gumbel approximation (Wang & Jegelka, section 3.1).

    Approximates the CDF of the maximum by the independent product
    ``P(y* < z) ~= prod_i Psi((z - mu_i) / sigma_i)`` over a space-filling grid, matches a
    Gumbel distribution to it at two quantiles, then samples by inverse transform.
    """
    grid = search_space.sample_normalized_params(n_grid_points, rng)
    with torch.no_grad():
        mean, var = gpr.posterior(torch.from_numpy(grid))
    sigma = torch.sqrt(var + 1e-12)

    def log_cdf_of_max(z: float) -> float:
        return float(torch.sum(torch.special.log_ndtr((z - mean) / sigma)))

    # Bracket the search. The upper end is generous so that log F(hi) is essentially zero.
    lo = float((mean - 5.0 * sigma).min())
    hi = float((mean + 5.0 * sigma).max())

    def quantile(target_q: float) -> float:
        log_target = math.log(target_q)
        left, right = lo, hi
        for _ in range(40):
            mid = 0.5 * (left + right)
            if log_cdf_of_max(mid) < log_target:
                left = mid
            else:
                right = mid
        return 0.5 * (left + right)

    z_lower = quantile(_GUMBEL_LOWER_Q)
    z_upper = quantile(_GUMBEL_UPPER_Q)

    # z_q = a - b * log(-log q), so b follows from the two quantile positions.
    scale = (z_upper - z_lower) / (_GUMBEL_LOWER_COEF - _GUMBEL_UPPER_COEF)
    scale = max(scale, 1e-8)
    location = z_lower + scale * _GUMBEL_LOWER_COEF

    uniforms = rng.uniform(1e-12, 1.0 - 1e-12, size=n_samples)
    samples = location - scale * np.log(-np.log(uniforms))
    return _clip_to_incumbent(samples, y_best, scale)


def _sample_max_values_by_posterior(
    gpr: gp.GPRegressor,
    search_space: gp_search_space.SearchSpace,
    n_samples: int,
    n_grid_points: int,
    rng: np.random.RandomState,
    y_best: float,
) -> torch.Tensor:
    """Sample maxima by drawing joint posterior paths (Wang & Jegelka, section 3.2).

    Draws ``n_samples`` joint realizations of the latent function on a set of representer
    points and takes the maximum of each. ``GPRegressor.posterior(..., joint=True)`` returns
    the full covariance, so this needs no random-Fourier-feature approximation.
    """
    grid = search_space.sample_normalized_params(n_grid_points, rng)
    with torch.no_grad():
        mean, cov = gpr.posterior(torch.from_numpy(grid), joint=True)
        # The covariance is PSD but can be near-singular on a dense grid; the jitter is
        # scaled to the problem so it stays negligible relative to the kernel amplitude.
        jitter = 1e-8 * float(torch.mean(torch.diagonal(cov)).clamp_min(1e-12))
        eye = torch.eye(cov.shape[-1], dtype=cov.dtype)
        for attempt in range(5):
            try:
                chol = torch.linalg.cholesky(cov + jitter * eye)
                break
            except Exception:
                jitter *= 100.0
        else:
            # Fall back to the marginals; correlated structure is lost but sampling proceeds.
            chol = torch.diag(torch.sqrt(torch.diagonal(cov).clamp_min(1e-12)))

        normals = torch.from_numpy(
            rng.standard_normal(size=(cov.shape[-1], n_samples))
        )  # (n_grid, n_samples)
        paths = mean.unsqueeze(-1) + chol @ normals
        samples = torch.max(paths, dim=0).values.numpy()

    return _clip_to_incumbent(samples, y_best, 1.0)


def _clip_to_incumbent(samples: np.ndarray, y_best: float, scale: float) -> torch.Tensor:
    """Keep sampled maxima strictly above the best observation.

    A max-value below the incumbent is inconsistent with the data and drives ``gamma``
    negative, where the acquisition is dominated by the ``-log Psi`` term and stops
    discriminating between candidates.
    """
    floor = y_best + 1e-3 * max(abs(scale), 1e-8)
    return torch.from_numpy(np.maximum(samples, floor))


class MESSampler(GPSampler):
    """Sampler using the max-value entropy search acquisition function.

    MES selects the candidate whose observation is expected to reduce the entropy of the
    distribution of the maximum value ``y*`` by the most. Unlike improvement-based
    acquisitions it has no incumbent threshold, and unlike UCB it has no exploration
    parameter to tune.

    Args:
        max_value_sampler:
            How to sample the maxima ``y*``. ``"posterior"`` (the default) draws joint
            posterior paths over the representer points as in Wang & Jegelka section 3.2,
            at the cost of one Cholesky factorization of an ``n_representer_points`` square
            matrix per trial. ``"gumbel"`` uses the Gumbel approximation of section 3.1,
            which needs only marginal posteriors and is cheaper, but treats the representer
            points as independent and so samples ``y*`` too high; see the README for the
            measured bias. On BBOB the cheaper variant did not earn its saving, which is why
            ``"posterior"`` is the default.
        n_max_value_samples: Number of maxima to average the acquisition over.
        n_representer_points: Size of the grid used to sample the maxima.
        seed: Random seed.
        independent_sampler: Sampler for parameters outside the intersection search space.
        n_startup_trials: Number of initial random trials before the GP is used.
        deterministic_objective: If :obj:`True`, assume the objective is noiseless.
        constraints_func: Constraint evaluation function.
        warn_independent_sampling: If :obj:`True`, warn when independent sampling is used.

    Note:
        Only single-objective, unconstrained optimization uses the MES acquisition function.
        Multi-objective and constrained studies fall back to the parent ``GPSampler``.
    """

    def __init__(
        self,
        *,
        max_value_sampler: str = "posterior",
        n_max_value_samples: int = 32,
        n_representer_points: int = 512,
        seed: int | None = None,
        independent_sampler: optuna.samplers.BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        warn_independent_sampling: bool = True,
    ) -> None:
        super().__init__(
            seed=seed,
            independent_sampler=independent_sampler,
            n_startup_trials=n_startup_trials,
            deterministic_objective=deterministic_objective,
            constraints_func=constraints_func,
            warn_independent_sampling=warn_independent_sampling,
        )
        if max_value_sampler not in ("gumbel", "posterior"):
            raise ValueError(
                f"max_value_sampler must be 'gumbel' or 'posterior', got {max_value_sampler!r}."
            )
        if n_max_value_samples < 1:
            raise ValueError(f"n_max_value_samples must be positive, got {n_max_value_samples}.")
        if n_representer_points < 1:
            raise ValueError(f"n_representer_points must be positive, got {n_representer_points}.")
        self._max_value_sampler = max_value_sampler
        self._n_max_value_samples = n_max_value_samples
        self._n_representer_points = n_representer_points

    def _create_acqf(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        standardized_score_vals: np.ndarray,
    ) -> acqf_module.BaseAcquisitionFunc:
        sample_max_values = (
            _sample_max_values_by_gumbel
            if self._max_value_sampler == "gumbel"
            else _sample_max_values_by_posterior
        )
        max_value_samples = sample_max_values(
            gpr,
            search_space,
            self._n_max_value_samples,
            self._n_representer_points,
            self._rng.rng,
            float(standardized_score_vals.max()),
        )
        return _MaxValueEntropySearch(
            gpr=gpr,
            search_space=search_space,
            max_value_samples=max_value_samples,
        )

    def _sample_relative_impl(
        self,
        study: Study,
        completed_trials: list[FrozenTrial],
        trials: list[FrozenTrial],
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        internal_search_space = gp_search_space.SearchSpace(search_space)
        normalized_params = internal_search_space.get_normalized_params(completed_trials)

        _sign = np.array([-1.0 if d == StudyDirection.MINIMIZE else 1.0 for d in study.directions])
        standardized_score_vals, _, _ = _standardize_values(
            _sign * np.array([trial.values for trial in completed_trials])
        )

        if (
            self._gprs_cache_list is not None  # type: ignore[has-type]
            and len(self._gprs_cache_list[0].inverse_squared_lengthscales)  # type: ignore[has-type]
            != internal_search_space.dim
        ):
            self._gprs_cache_list = None

        n_objectives = standardized_score_vals.shape[-1]

        # Multi-objective and constrained cases fall back to the parent GPSampler.
        if n_objectives > 1 or self._constraints_func is not None:
            return super()._sample_relative_impl(study, completed_trials, trials, search_space)

        cache = self._gprs_cache_list[0] if self._gprs_cache_list is not None else None  # type: ignore[index]
        gpr_obj = gp.fit_kernel_params(
            X=normalized_params,
            Y=standardized_score_vals[:, 0],
            is_categorical=internal_search_space.is_categorical,
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            gpr_cache=cache,
            deterministic_objective=self._deterministic,
        )
        self._gprs_cache_list = [gpr_obj]

        acqf = self._create_acqf(
            gpr=gpr_obj,
            search_space=internal_search_space,
            standardized_score_vals=standardized_score_vals[:, 0],
        )
        best_params = normalized_params[np.argmax(standardized_score_vals[:, 0]), np.newaxis]

        normalized_param = self._optimize_acqf(acqf, best_params)
        return internal_search_space.get_unnormalized_param(normalized_param)
