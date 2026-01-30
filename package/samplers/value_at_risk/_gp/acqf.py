from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import math
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from typing import Protocol

    from optuna._gp.search_space import SearchSpace
    import torch

    from .gp import GPRegressor

    class SobolGenerator(Protocol):
        def __call__(self, dim: int, n_samples: int, seed: int | None) -> torch.Tensor:
            raise NotImplementedError

else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


_EPS = 1e-12  # NOTE(nabenabe): grad becomes nan when EPS=0.


def _sample_from_sobol(dim: int, n_samples: int, seed: int | None) -> torch.Tensor:
    return torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
        dimension=dim, scramble=True, seed=seed
    ).draw(n_samples, dtype=torch.float64)


def _sample_from_normal_sobol(dim: int, n_samples: int, seed: int | None) -> torch.Tensor:
    # NOTE(nabenabe): Normal Sobol sampling based on BoTorch.
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/sampling/qmc.py#L26-L97
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/sampling.py#L109-L138
    sobol_samples = _sample_from_sobol(dim, n_samples, seed)
    samples = 2.0 * (sobol_samples - 0.5)  # The Sobol sequence in [-1, 1].
    # Inverse transform to standard normal (values to close to -1 or 1 result in infinity).
    return torch.erfinv(samples) * float(np.sqrt(2))


def _sample_input_noise(
    n_input_noise_samples: int,
    uniform_input_noise_rads: torch.Tensor | None,
    normal_input_noise_stdevs: torch.Tensor | None,
    seed: int | None,
) -> torch.Tensor:
    assert uniform_input_noise_rads is not None or normal_input_noise_stdevs is not None
    if normal_input_noise_stdevs is not None:
        dim = normal_input_noise_stdevs.size(0)
        noisy_inds = torch.where(normal_input_noise_stdevs != 0.0)
        input_noise = torch.zeros(size=(n_input_noise_samples, dim), dtype=torch.float64)
        input_noise[:, noisy_inds[0]] = (
            _sample_from_normal_sobol(noisy_inds[0].size(0), n_input_noise_samples, seed)
            * normal_input_noise_stdevs[noisy_inds]
        )
        return input_noise
    elif uniform_input_noise_rads is not None:
        dim = uniform_input_noise_rads.size(0)
        noisy_inds = torch.where(uniform_input_noise_rads != 0.0)
        input_noise = torch.zeros(size=(n_input_noise_samples, dim), dtype=torch.float64)
        input_noise[:, noisy_inds[0]] = (
            _sample_from_sobol(noisy_inds[0].size(0), n_input_noise_samples, seed)
            * 2
            * uniform_input_noise_rads[noisy_inds]
            - uniform_input_noise_rads[noisy_inds]
        )
        return input_noise
    else:
        raise ValueError(
            "Either `uniform_input_noise_rads` or `normal_input_noise_stdevs` must be provided."
        )


class BaseAcquisitionFunc(ABC):
    def __init__(self, length_scales: np.ndarray, search_space: SearchSpace) -> None:
        self.length_scales = length_scales
        self.search_space = search_space

    @abstractmethod
    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def eval_acqf_no_grad(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.eval_acqf(torch.from_numpy(x)).detach().numpy()

    def eval_acqf_with_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        assert x.ndim == 1
        x_tensor = torch.from_numpy(x).requires_grad_(True)
        val = self.eval_acqf(x_tensor)
        val.backward()  # type: ignore
        return val.item(), x_tensor.grad.detach().numpy()  # type: ignore


class LogCumulativeProbabilityAtRisk(BaseAcquisitionFunc):
    """The logarithm of the cumulative probability measure at risk

    When we replace ``f(x)`` in VaR with ``1[f(x) <= f*]``, the optimization of the new VaR corresponds to
    that of the mean probability of ``x`` with input perturbation being feasible.
    """

    def __init__(
        self,
        gpr_list: list[GPRegressor],
        search_space: SearchSpace,
        confidence_level: float,
        threshold_list: list[float],
        n_input_noise_samples: int,
        qmc_seed: int | None,
        fixed_indices: torch.Tensor,
        fixed_values: torch.Tensor,
        uniform_input_noise_rads: torch.Tensor | None = None,
        normal_input_noise_stdevs: torch.Tensor | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr_list = gpr_list
        self._threshold_list = threshold_list
        rng = np.random.RandomState(qmc_seed)
        self._input_noise = _sample_input_noise(
            n_input_noise_samples,
            uniform_input_noise_rads,
            normal_input_noise_stdevs,
            seed=rng.random_integers(0, 2**31 - 1, size=1).item(),
        )
        self._stabilizing_noise = stabilizing_noise
        self._confidence_level = confidence_level
        self._fixed_indices = fixed_indices
        self._fixed_values = fixed_values
        super().__init__(
            length_scales=np.mean([gpr.length_scales for gpr in gpr_list], axis=0),
            search_space=search_space,
        )

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        # The search space of constant noisy parameters is internally replaced with IntDistribution(0, 0),
        # so that their normalized values passed as x is always 0.5. However, values passed to
        # const_noisy_param_values argument of RobustGPSampler._get_value_at_risk may be normalized to
        # different values under the original search space used by GPRegressor.  So we carry around the
        # normalized version of const_noisy_param_values explicity and use them instead.
        x = x.clone()
        x[:, self._fixed_indices] = self._fixed_values

        x_noisy = x.unsqueeze(-2) + self._input_noise
        log_feas_probs = torch.zeros(x_noisy.shape[:-1], dtype=torch.float64)
        for gpr, threshold in zip(self._gpr_list, self._threshold_list):
            means, vars_ = gpr.posterior(x_noisy)
            sigmas = torch.sqrt(vars_ + self._stabilizing_noise)
            # NOTE(nabenabe): integral from a to b of f(x) is integral from -b to -a of f(-x).
            log_feas_probs += torch.special.log_ndtr((means - threshold) / sigmas)
        n_input_noise_samples = len(self._input_noise)
        n_risky_samples = max(1, math.ceil((1 - self._confidence_level) * n_input_noise_samples))
        log_feas_probs_at_risk, _ = torch.topk(
            log_feas_probs,
            k=n_risky_samples,
            dim=-1,
            largest=False,
            sorted=False,
        )
        return log_feas_probs_at_risk.logsumexp(dim=-1) - math.log(n_risky_samples)


class ValueAtRisk(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        confidence_level: float,
        n_input_noise_samples: int,
        n_qmc_samples: int,
        qmc_seed: int | None,
        acqf_type: str,
        fixed_indices: torch.Tensor,
        fixed_values: torch.Tensor,
        uniform_input_noise_rads: torch.Tensor | None = None,
        normal_input_noise_stdevs: torch.Tensor | None = None,
    ) -> None:
        assert 0 <= confidence_level <= 1
        self._gpr = gpr
        self._confidence_level = confidence_level
        self._rng = np.random.RandomState(qmc_seed)
        self._input_noise = _sample_input_noise(
            n_input_noise_samples,
            uniform_input_noise_rads,
            normal_input_noise_stdevs,
            seed=self._rng.random_integers(0, 2**31 - 1, size=1).item(),
        )
        self._fixed_samples = _sample_from_normal_sobol(
            dim=n_input_noise_samples,
            n_samples=n_qmc_samples,
            seed=self._rng.random_integers(0, 2**31 - 1, size=1).item(),
        )
        self._acqf_type = acqf_type
        self._robust_X_noisy: torch.Tensor
        self._covar_robust_chol: torch.Tensor
        self._var_thresholds: torch.Tensor
        self._V: torch.Tensor
        self._S1: torch.Tensor
        self._S2: torch.Tensor
        self._fixed_indices = fixed_indices
        self._fixed_values = fixed_values
        super().__init__(length_scales=gpr.length_scales, search_space=search_space)

    def set_robust_X_noisy(self, X: torch.Tensor, n_samples: int) -> None:
        n_qmc_samples = self._fixed_samples.shape[0]
        self._fixed_samples = _sample_from_normal_sobol(
            dim=self._input_noise.shape[0],
            n_samples=n_samples,
            seed=self._rng.random_integers(0, 2**31 - 1, size=1).item(),
        )
        robust_X = X[self._value_at_risk(X).argmax(dim=0).unique()]
        robust_X_noisy = robust_X.unsqueeze(-2) + self._input_noise
        self._robust_X_noisy = robust_X_noisy.view(-1, robust_X_noisy.size(-1))
        means_robust, covar_robust = self._gpr.posterior(self._robust_X_noisy, joint=True)
        self._covar_robust_chol = torch.linalg.cholesky(covar_robust)
        fixed_samples = _sample_from_normal_sobol(
            dim=self._input_noise.shape[0] * (robust_X.shape[0] + 1),
            n_samples=n_qmc_samples,
            seed=self._rng.random_integers(0, 2**31 - 1, size=1).item(),
        )
        cov_fX_fX1 = self._gpr.kernel(self._gpr._X_train, self._robust_X_noisy)
        # inv(C) @ K = V --> K = C @ V --> K = L @ L.T @ V
        cov_Y_Y_chol = self._gpr._cov_Y_Y_chol
        assert cov_Y_Y_chol is not None
        self._V = torch.linalg.solve_triangular(
            cov_Y_Y_chol.T,
            torch.linalg.solve_triangular(cov_Y_Y_chol, cov_fX_fX1, upper=False),
            upper=True,
        )
        self._S1 = fixed_samples[:, : self._robust_X_noisy.shape[0]]
        self._S2 = fixed_samples[:, self._robust_X_noisy.shape[0] :]
        posterior_at_X_noisy = (
            means_robust.unsqueeze(-2) + self._S1 @ self._covar_robust_chol
        ).view(n_qmc_samples, *robust_X_noisy.shape[:-1])
        var_at_X_noisy = torch.quantile(posterior_at_X_noisy, q=self._confidence_level, dim=-1)
        self._var_thresholds = var_at_X_noisy.amax(dim=-1)

    def _value_at_risk(self, x: torch.Tensor) -> torch.Tensor:
        means, covar = self._gpr.posterior(x.unsqueeze(-2) + self._input_noise, joint=True)
        L = torch.linalg.cholesky(covar)
        posterior_samples = means.unsqueeze(-2) + self._fixed_samples.matmul(L)
        # If CVaR, use torch.topk instead of torch.quantile.
        return torch.quantile(posterior_samples, q=self._confidence_level, dim=-1)

    def posterior_covar_cross_block(self, X: torch.Tensor) -> torch.Tensor:
        cov_fX2_fX1 = self._gpr.kernel(X, self._robust_X_noisy)
        cov_fX2_fX = self._gpr.kernel(X, self._gpr._X_train)
        return cov_fX2_fX1 - cov_fX2_fX.matmul(self._V)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denote ``cov11`` as the covariance matrix at ``robust_X_noisy``, ``cov22`` as that at ``x_noisy``,
        ``cov12`` as the cross covariance matrix between ``robust_X_noisy`` and ``x_noisy``.
        Consider the block matrix ``cov = [[cov11, cov12], [cov21, cov22]]`` and its Cholesky
        decomposition ``L = [[L11, 0], [L21, L22]]``.
        Since ``L @ L.T = cov``::

            [[L11, 0], [L21, L22]] @ [[L11.T, L21.T], [0, L22.T]]
            = [[L11 @ L11.T, L11 @ L21.T], [L21 @ L11.T, L21 @ L21.T + L22 @ L22.T]]
            = cov

        Thus, we have ``L11 = chol(cov11)``, ``L21 = cov21 @ inv(L11.T)``, and
        ``L22 = chol(cov22 - L21 @ L21.T)``. Note that ``L21 = cov21 @ inv(L11.T)``
        implies ``L21 @ L11.T = cov21``, so we can solve the triangular to yield ``L21``.

        Let's divide a fixed sample into ``S = [S1, S2]`` then ``S @ L = [L11 @ S1, L21 @ S1 + L22 @ S2]``.
        We only need to compute ``L21 @ S1 + L22 @ S2``, which is the posterior sampling at ``x_noisy``.
        """

        # The search space of constant noisy parameters is internally replaced with IntDistribution(0, 0),
        # so that their normalized values passed as x is always 0.5. However, values passed to
        # const_noisy_param_values argument of RobustGPSampler._get_value_at_risk may be normalized to
        # different values under the original search space used by GPRegressor.  So we carry around the
        # normalized version of const_noisy_param_values explicity and use them instead.
        x = x.clone()
        x[:, self._fixed_indices] = self._fixed_values

        if self._acqf_type == "mean":
            return self._value_at_risk(x).mean(dim=-1)

        x_noisy = x.unsqueeze(-2) + self._input_noise
        means, covar = self._gpr.posterior(x_noisy, joint=True)
        cov21 = self.posterior_covar_cross_block(x_noisy)
        L21 = torch.linalg.solve_triangular(
            self._covar_robust_chol.transpose(-1, -2), cov21, left=False, upper=True
        )
        # Add a small jitter to prevent the matrix from becoming non-positive definite due to numerical issues.
        # This issue appears to happen when the x is included in robust_X.
        L22 = torch.linalg.cholesky(
            covar - L21.matmul(L21.transpose(-1, -2)) + 1e-12 * torch.eye(covar.shape[-1])
        )
        posterior_at_x_noisy = (
            means.unsqueeze(-2)
            + torch.einsum("...ij,kj->...ki", L21, self._S1)
            + self._S2.matmul(L22)
        )
        var_at_x_noisy = torch.quantile(posterior_at_x_noisy, q=self._confidence_level, dim=-1)
        return (var_at_x_noisy - self._var_thresholds).clamp_min(_EPS).mean(dim=-1)


class ConstrainedLogValueAtRisk(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        constraints_gpr_list: list[GPRegressor],
        constraints_threshold_list: list[float],
        objective_confidence_level: float,
        feas_prob_confidence_level: float,
        n_input_noise_samples: int,
        n_qmc_samples: int,
        qmc_seed: int | None,
        acqf_type: str,
        fixed_indices: torch.Tensor,
        fixed_values: torch.Tensor,
        uniform_input_noise_rads: torch.Tensor | None = None,
        normal_input_noise_stdevs: torch.Tensor | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._value_at_risk = ValueAtRisk(
            gpr=gpr,
            search_space=search_space,
            confidence_level=objective_confidence_level,
            n_input_noise_samples=n_input_noise_samples,
            n_qmc_samples=n_qmc_samples,
            qmc_seed=qmc_seed,
            acqf_type=acqf_type,
            uniform_input_noise_rads=uniform_input_noise_rads,
            normal_input_noise_stdevs=normal_input_noise_stdevs,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
        )
        self._acqf_type = acqf_type
        if acqf_type == "nei":
            self._value_at_risk.set_robust_X_noisy(gpr._X_train, n_samples=128)

        self._log_prob_at_risk = LogCumulativeProbabilityAtRisk(
            gpr_list=constraints_gpr_list,
            search_space=search_space,
            confidence_level=feas_prob_confidence_level,
            threshold_list=constraints_threshold_list,
            n_input_noise_samples=n_input_noise_samples,
            qmc_seed=qmc_seed,
            uniform_input_noise_rads=uniform_input_noise_rads,
            normal_input_noise_stdevs=normal_input_noise_stdevs,
            stabilizing_noise=stabilizing_noise,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
        )
        assert torch.allclose(
            self._log_prob_at_risk._input_noise, self._value_at_risk._input_noise
        )
        super().__init__(self._value_at_risk.length_scales, search_space=search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        out = self._log_prob_at_risk.eval_acqf(x)
        if self._acqf_type == "mean":
            out += self._value_at_risk.eval_acqf(x).clamp_min(_EPS).log_()
        else:
            out += self._value_at_risk.eval_acqf(x).log_()
        return out
