"""GP-based samplers with alternative acquisition functions (PI, UCB, TS).

This package provides GP-based Bayesian optimization samplers with acquisition functions
beyond the default Expected Improvement (EI) in Optuna's GPSampler.

See https://github.com/optuna/optunahub-registry/issues/119 for details.
"""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

import optuna
from optuna.samplers._gp.sampler import _standardize_values
from optuna.samplers._gp.sampler import GPSampler
from optuna.study import StudyDirection


if TYPE_CHECKING:
    import torch

    import optuna._gp.acqf as acqf_module
    import optuna._gp.gp as gp
    import optuna._gp.search_space as gp_search_space
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
    gp_search_space = _LazyImport("optuna._gp.search_space")
    gp = _LazyImport("optuna._gp.gp")
    acqf_module = _LazyImport("optuna._gp.acqf")


__all__ = ["GPEISampler", "GPPISampler", "GPUCBSampler", "GPTSSampler"]


class _GPAcqfSamplerBase(GPSampler):
    """Base class for GP samplers with custom acquisition functions.

    Subclasses override ``_create_acqf`` to plug in a different acquisition function
    while reusing all of GPSampler's GP fitting, search-space handling, and acqf
    optimization machinery.

    Note:
        Only single-objective, unconstrained optimization is supported for the
        alternative acquisition functions (PI, UCB, TS). Multi-objective and
        constrained setups fall back to the parent GPSampler behaviour (logEI /
        logEHVI / constrained variants).
    """

    def _create_acqf(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        standardized_score_vals: np.ndarray,
    ) -> acqf_module.BaseAcquisitionFunc:
        raise NotImplementedError

    def _sample_relative_impl(
        self,
        study: Study,
        completed_trials: list[FrozenTrial],
        trials: list[FrozenTrial],
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        internal_search_space = gp_search_space.SearchSpace(search_space)
        normalized_params = internal_search_space.get_normalized_params(completed_trials)

        _sign = np.array(
            [-1.0 if d == StudyDirection.MINIMIZE else 1.0 for d in study.directions]
        )
        standardized_score_vals, _, _ = _standardize_values(
            _sign * np.array([trial.values for trial in completed_trials])
        )

        if (
            self._gprs_cache_list is not None
            and len(self._gprs_cache_list[0].inverse_squared_lengthscales)
            != internal_search_space.dim
        ):
            self._gprs_cache_list = None

        n_objectives = standardized_score_vals.shape[-1]

        # For multi-objective or constrained cases, fall back to parent GPSampler.
        if n_objectives > 1 or self._constraints_func is not None:
            return super()._sample_relative_impl(
                study, completed_trials, trials, search_space
            )

        is_categorical = internal_search_space.is_categorical
        cache = self._gprs_cache_list[0] if self._gprs_cache_list is not None else None
        gpr_obj = gp.fit_kernel_params(
            X=normalized_params,
            Y=standardized_score_vals[:, 0],
            is_categorical=is_categorical,
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
        best_params = normalized_params[
            np.argmax(standardized_score_vals[:, 0]), np.newaxis
        ]

        normalized_param = self._optimize_acqf(acqf, best_params)
        return internal_search_space.get_unnormalized_param(normalized_param)


# --------------------------------------------------------------------------- #
#  GPEISampler -- alias for GPSampler (as specified in the issue)
# --------------------------------------------------------------------------- #

GPEISampler = GPSampler


# --------------------------------------------------------------------------- #
#  GPPISampler -- Probability of Improvement
# --------------------------------------------------------------------------- #


class GPPISampler(_GPAcqfSamplerBase):
    """GP-based sampler using Probability of Improvement (PI) acquisition function.

    PI selects the point with the highest probability of improving over the current
    best observation. It tends to be more exploitative than EI.

    Args:
        seed: Random seed.
        independent_sampler: Sampler for independent parameters.
        n_startup_trials: Number of initial random trials before GP kicks in.
        deterministic_objective: If ``True``, assume the objective is noiseless.
    """

    def _create_acqf(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        standardized_score_vals: np.ndarray,
    ) -> acqf_module.BaseAcquisitionFunc:
        return acqf_module.LogPI(
            gpr=gpr,
            search_space=search_space,
            threshold=float(standardized_score_vals.max()),
        )


# --------------------------------------------------------------------------- #
#  GPUCBSampler -- Upper Confidence Bound
# --------------------------------------------------------------------------- #


class GPUCBSampler(_GPAcqfSamplerBase):
    """GP-based sampler using Upper Confidence Bound (UCB) acquisition function.

    UCB balances exploration and exploitation via a ``beta`` parameter that controls
    the width of the confidence interval. Larger ``beta`` encourages more exploration.

    The default ``beta=2.0`` corresponds to a ~95% confidence bound, which is a
    common choice in the literature (Srinivas et al., 2010).

    Args:
        beta: Exploration-exploitation trade-off parameter (default: 2.0).
        seed: Random seed.
        independent_sampler: Sampler for independent parameters.
        n_startup_trials: Number of initial random trials before GP kicks in.
        deterministic_objective: If ``True``, assume the objective is noiseless.
    """

    def __init__(
        self,
        *,
        beta: float = 2.0,
        seed: int | None = None,
        independent_sampler: optuna.samplers.BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
    ) -> None:
        super().__init__(
            seed=seed,
            independent_sampler=independent_sampler,
            n_startup_trials=n_startup_trials,
            deterministic_objective=deterministic_objective,
        )
        self._beta = beta

    def _create_acqf(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        standardized_score_vals: np.ndarray,
    ) -> acqf_module.BaseAcquisitionFunc:
        return acqf_module.UCB(
            gpr=gpr,
            search_space=search_space,
            beta=self._beta,
        )


# --------------------------------------------------------------------------- #
#  GPTSSampler -- Thompson Sampling via Random Fourier Features
# --------------------------------------------------------------------------- #


class _ThompsonSampling(acqf_module.BaseAcquisitionFunc):
    """Thompson Sampling acquisition function using random Fourier features.

    Draws a single sample function from the GP posterior using random Fourier
    features (RFF) approximation and returns its value as the acquisition score.
    This provides a principled exploration-exploitation trade-off without any
    tuning parameter.

    Reference:
        Hernandez-Lobato et al., "Predictive Entropy Search for Efficient Global
        Optimization of Black-box Functions" (2014).
        Rahimi & Recht, "Random Features for Large-Scale Kernel Machines" (2007).
    """

    def __init__(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        seed: int | None = None,
        n_features: int = 512,
    ) -> None:
        self._gpr = gpr
        self._seed = seed
        self._n_features = n_features

        # RFF state (lazily initialized on first eval_acqf call).
        self._theta: torch.Tensor | None = None
        self._rff_weights: torch.Tensor | None = None
        self._rff_bias: torch.Tensor | None = None

        super().__init__(gpr.length_scales, search_space)

    def _setup_rff(self, dim: int) -> None:
        """Set up Random Fourier Features for Matern 5/2 kernel approximation."""
        generator = torch.Generator()
        if self._seed is not None:
            generator.manual_seed(self._seed)
        else:
            generator.seed()

        length_scales = torch.from_numpy(self._gpr.length_scales)

        # Matern 5/2 spectral density is Student-t with nu=5 degrees of freedom.
        # Sample via normal / sqrt(chi2/nu) ratio.
        nu = 5.0
        normal_samples = torch.randn(
            self._n_features, dim, generator=generator, dtype=torch.float64
        )
        chi2_samples = torch.sum(
            torch.randn(
                self._n_features, int(nu), generator=generator, dtype=torch.float64
            )
            ** 2,
            dim=-1,
            keepdim=True,
        )
        spectral_samples = normal_samples / torch.sqrt(chi2_samples / nu)

        # Scale by inverse lengthscales.
        self._rff_weights = (spectral_samples / length_scales.unsqueeze(0)).detach()
        self._rff_bias = (
            torch.rand(self._n_features, generator=generator, dtype=torch.float64)
            * 2
            * np.pi
        ).detach()

        # Compute RFF features for training data.
        X_train = self._gpr._X_train
        Y_train = self._gpr._y_train
        Phi_train = self._rff_feature(X_train)  # (n_train, 2 * n_features)

        # Bayesian linear regression posterior.
        noise_var = float(self._gpr.noise_var)
        kernel_scale = float(self._gpr.kernel_scale)

        n_rff = Phi_train.shape[1]
        A = (
            Phi_train.T @ Phi_train / noise_var
            + torch.eye(n_rff, dtype=torch.float64) / kernel_scale
        )
        b = Phi_train.T @ Y_train / noise_var

        # Posterior mean: A^{-1} b
        L = torch.linalg.cholesky(A)
        theta_mean = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)

        # Sample theta ~ N(theta_mean, A^{-1})
        z = torch.randn(n_rff, generator=generator, dtype=torch.float64)
        theta_sample = theta_mean + torch.linalg.solve_triangular(
            L.T, z.unsqueeze(-1), upper=True
        ).squeeze(-1)

        self._theta = theta_sample.detach()

    def _rff_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Random Fourier Features: sqrt(2/D) * [cos(Wx+b), sin(Wx+b)]."""
        assert self._rff_weights is not None and self._rff_bias is not None
        proj = x @ self._rff_weights.T + self._rff_bias  # (..., n_features)
        scale = np.sqrt(2.0 / self._n_features)
        return scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        if self._theta is None:
            self._setup_rff(x.shape[-1])
        assert self._theta is not None
        Phi = self._rff_feature(x)  # (..., 2 * n_features)
        return Phi @ self._theta  # (...,)


class GPTSSampler(_GPAcqfSamplerBase):
    """GP-based sampler using Thompson Sampling (TS) acquisition function.

    Thompson Sampling draws a sample function from the GP posterior and selects
    the point that maximizes this sample. This provides a natural balance between
    exploration and exploitation without any tuning parameter.

    The posterior sample is approximated via Random Fourier Features (RFF) for
    computational efficiency.

    Args:
        n_rff_features: Number of random Fourier features for posterior
            approximation (default: 512).
        seed: Random seed.
        independent_sampler: Sampler for independent parameters.
        n_startup_trials: Number of initial random trials before GP kicks in.
        deterministic_objective: If ``True``, assume the objective is noiseless.
    """

    def __init__(
        self,
        *,
        n_rff_features: int = 512,
        seed: int | None = None,
        independent_sampler: optuna.samplers.BaseSampler | None = None,
        n_startup_trials: int = 10,
        deterministic_objective: bool = False,
    ) -> None:
        super().__init__(
            seed=seed,
            independent_sampler=independent_sampler,
            n_startup_trials=n_startup_trials,
            deterministic_objective=deterministic_objective,
        )
        self._n_rff_features = n_rff_features

    def _create_acqf(
        self,
        gpr: gp.GPRegressor,
        search_space: gp_search_space.SearchSpace,
        standardized_score_vals: np.ndarray,
    ) -> acqf_module.BaseAcquisitionFunc:
        return _ThompsonSampling(
            gpr=gpr,
            search_space=search_space,
            seed=self._rng.rng.randint(1 << 30),
            n_features=self._n_rff_features,
        )
