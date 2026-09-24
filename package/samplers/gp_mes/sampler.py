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
from optuna.samplers._gp.sampler import _get_params
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


__all__ = ["GIBBONSampler", "MESSampler"]

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


class _GIBBON(acqf_module.BaseAcquisitionFunc):
    """The GIBBON lower bound on the batch information gain about ``y*``.

    Moss, Leslie, Gonzalez & Rayson, "GIBBON: General-purpose Information-Based Bayesian
    OptimisatioN", JMLR 22(235), 2021, eq. (7). Writing ``R`` for the posterior correlation
    matrix of the batch, ``gamma_i = (y* - mu_i) / sigma_i``, and ``phi``, ``Psi`` for the
    standard normal PDF and CDF, the bound for one sampled maximum is

        IG(batch, y*) = 0.5 log|R| - 0.5 sum_i log(1 - rho_i^2 r_i (gamma_i + r_i))

    with ``r_i = phi(gamma_i) / Psi(gamma_i)`` and ``gamma_i = (y* - mu_i) / sigma_i``. The
    paper distinguishes the observations from the latent function they inform: ``R`` is the
    correlation matrix of the *observations*, whose covariance carries the fitted noise,
    while ``gamma`` is formed from the *latent* posterior and
    ``rho_i^2 = latent_var_i / observation_var_i`` is the squared correlation between the
    two. They coincide only for a noiseless objective, which is not Optuna's default.

    The first term rewards batches whose members are weakly correlated, and is what makes
    this a batch acquisition rather than a per-point one. The second is a per-point
    information term closely related to max-value entropy search: at batch size one the
    first term vanishes and the bound sits strictly below Wang & Jegelka's exact value.

    Optuna evaluates one trial at a time, so the batch here is the set of currently running
    trials together with the candidate under consideration. Scoring a candidate therefore
    penalises it for resembling work already in flight, which is the diversity mechanism.
    """

    def __init__(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        max_value_samples: torch.Tensor,
        pending: np.ndarray | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr = gpr
        self._max_value_samples = max_value_samples
        self._stabilizing_noise = stabilizing_noise
        self._noise_var = float(gpr.noise_var)
        self._pending = None if pending is None or len(pending) == 0 else torch.from_numpy(pending)

        self._pending_logdet_corr = torch.zeros((), dtype=torch.float64)
        self._pending_info = torch.zeros((), dtype=torch.float64)
        if self._pending is not None:
            mean_p, cov_latent = gpr.posterior(self._pending, joint=True)
            # eq. (7) is stated for the observations A, whose covariance carries the
            # observation noise; gamma is formed from the latent C. They coincide only when
            # the objective is deterministic, which is not Optuna's default.
            cov_p = cov_latent + self._noise_var * torch.eye(
                cov_latent.shape[-1], dtype=cov_latent.dtype
            )
            # Pending trials may coincide exactly, which categorical parameters make
            # common. The observation noise on the diagonal already keeps the matrix
            # invertible in that case; this jitter is a guard for the degenerate
            # `deterministic_objective` setting, where the noise floor is 1e-6.
            jitter = 1e-10 * float(torch.diagonal(cov_p).mean().clamp_min(1e-12))
            cov_p = cov_p + jitter * torch.eye(cov_p.shape[-1], dtype=cov_p.dtype)
            sd_p = torch.sqrt(torch.diagonal(cov_p))
            corr_p = cov_p / torch.outer(sd_p, sd_p)
            self._pending_cov_chol = torch.linalg.cholesky(cov_p)
            self._pending_logdet_corr = torch.linalg.slogdet(corr_p)[1]
            latent_sd_p = torch.sqrt(torch.diagonal(cov_latent) + stabilizing_noise)
            self._pending_info = self._log_reduction(
                mean_p, latent_sd_p, torch.diagonal(cov_latent) / torch.diagonal(cov_p)
            ).sum(dim=-1)
            # Depends only on the pending set and the fitted GP, so it is computed once.
            self._u_pending = torch.linalg.solve_triangular(
                gpr._cov_Y_Y_chol, gpr.kernel(gpr._X_all, self._pending), upper=False
            )

        super().__init__(gpr.length_scales, search_space)

    def _log_reduction(
        self, mean: torch.Tensor, latent_sd: torch.Tensor, rho_sq: torch.Tensor
    ) -> torch.Tensor:
        """log(1 - rho^2 r (gamma + r)) per point. Shape (n_samples, *mean.shape)."""
        y_star = self._max_value_samples.reshape((-1,) + (1,) * mean.ndim)
        gamma = (y_star - mean) / latent_sd
        log_cdf = torch.special.log_ndtr(gamma)
        log_pdf = -0.5 * gamma**2 - _LOG_SQRT_2PI
        ratio = torch.exp(log_pdf - log_cdf)  # phi/Psi, formed in log space for the tail
        inner = 1.0 - rho_sq * ratio * (gamma + ratio)
        # The bracket is a variance ratio and so lies in (0, 1]; the clamp guards rounding.
        return torch.log(torch.clamp(inner, min=1e-300))

    def _cross_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior covariance between the pending points and each candidate."""
        assert self._pending is not None
        u_x = torch.linalg.solve_triangular(
            self._gpr._cov_Y_Y_chol, self._gpr.kernel(self._gpr._X_all, x), upper=False
        )
        return self._gpr.kernel(self._pending, x) - self._u_pending.T @ u_x

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, latent_var = self._gpr.posterior(x)
        latent_var = latent_var + self._stabilizing_noise
        obs_var = latent_var + self._noise_var
        own = self._log_reduction(mean, torch.sqrt(latent_var), latent_var / obs_var)

        if self._pending is None:
            return torch.mean(-0.5 * own, dim=0)

        # log|R| for {pending, x} via the bordered determinant: the Schur complement is the
        # only part that varies with the candidate, so one solve serves every candidate.
        flat = x.reshape(-1, x.shape[-1])
        cross = self._cross_covariance(flat)
        solved = torch.cholesky_solve(cross, self._pending_cov_chol)
        obs_flat = obs_var.reshape(-1)
        schur = obs_flat - (cross * solved).sum(dim=0)
        # Relative floor, matching the relative jitter applied to the pending covariance.
        schur = torch.clamp(schur, min=self._stabilizing_noise * obs_flat)
        logdet_corr = (self._pending_logdet_corr + torch.log(schur / obs_flat)).reshape(
            obs_var.shape
        )

        info = self._pending_info.reshape((-1,) + (1,) * mean.ndim) + own
        return torch.mean(0.5 * logdet_corr - 0.5 * info, dim=0)


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

    #: Whether ``_create_acqf`` consumes the pending set. Read-only, so it carries none of
    #: the thread-safety hazard that motivated passing ``pending`` by argument.
    _uses_pending_trials = False

    def _sample_max_values(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        standardized_score_vals: np.ndarray,
    ) -> torch.Tensor:
        sample = (
            _sample_max_values_by_gumbel
            if self._max_value_sampler == "gumbel"
            else _sample_max_values_by_posterior
        )
        return sample(
            gpr,
            search_space,
            self._n_max_value_samples,
            self._n_representer_points,
            self._rng.rng,
            float(standardized_score_vals.max()),
        )

    def _create_acqf(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        standardized_score_vals: np.ndarray,
        pending: np.ndarray | None = None,
    ) -> acqf_module.BaseAcquisitionFunc:
        return _MaxValueEntropySearch(
            gpr=gpr,
            search_space=search_space,
            max_value_samples=self._sample_max_values(gpr, search_space, standardized_score_vals),
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

        # Passed by argument rather than stored on the sampler: one instance is shared
        # across threads under `n_jobs > 1`, and the kernel fit above releases the GIL long
        # enough that instance state is overwritten by another thread before it is read.
        # Derived only when the acquisition will use it, since `_get_params` parses the
        # system attributes of every running trial.
        pending = (
            internal_search_space.get_normalized_params(trials, [_get_params(t) for t in trials])
            if trials and self._uses_pending_trials
            else None
        )
        acqf = self._create_acqf(
            gpr=gpr_obj,
            search_space=internal_search_space,
            standardized_score_vals=standardized_score_vals[:, 0],
            pending=pending,
        )
        best_params = normalized_params[np.argmax(standardized_score_vals[:, 0]), np.newaxis]

        normalized_param = self._optimize_acqf(acqf, best_params)
        return internal_search_space.get_unnormalized_param(normalized_param)


class GIBBONSampler(MESSampler):
    """Batch-aware max-value entropy search, using the GIBBON lower bound.

    Extends :class:`MESSampler` with the acquisition of Moss, Leslie, Gonzalez & Rayson
    (JMLR 2021), which scores a *set* of evaluations rather than a single point. The extra
    term rewards batches whose members are weakly correlated under the GP posterior, so a
    candidate resembling work already in flight is penalised.

    Optuna runs one trial at a time, so the batch is taken to be the currently running
    trials plus the candidate under consideration. **This sampler is only worth using when
    trials run concurrently** -- through ``n_jobs``, multiple workers, or an ask-and-tell
    loop. With no running trials the correlation term vanishes and the acquisition reduces
    to a lower bound on the quantity :class:`MESSampler` already computes exactly, so
    :class:`MESSampler` is the better choice for sequential studies.

    Args:
        Identical to :class:`MESSampler`.

    Reference:
        Henry B. Moss, David S. Leslie, Javier Gonzalez and Paul Rayson. GIBBON:
        General-purpose Information-Based Bayesian OptimisatioN. Journal of Machine
        Learning Research, 22(235):1-49, 2021.
    """

    _uses_pending_trials = True

    def _create_acqf(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        standardized_score_vals: np.ndarray,
        pending: np.ndarray | None = None,
    ) -> acqf_module.BaseAcquisitionFunc:
        return _GIBBON(
            gpr=gpr,
            search_space=search_space,
            max_value_samples=self._sample_max_values(gpr, search_space, standardized_score_vals),
            pending=pending,
        )
