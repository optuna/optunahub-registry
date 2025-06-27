from __future__ import annotations

import math
from typing import Any
from typing import TYPE_CHECKING

import numpy as np
import scipy.optimize as so
import torch


if TYPE_CHECKING:
    from collections.abc import Callable


def convert_inf(values: np.ndarray) -> np.ndarray:
    is_values_finite = np.isfinite(values)
    if np.all(is_values_finite):
        return values

    is_any_finite = np.any(is_values_finite, axis=0)
    return np.clip(
        values,
        np.where(is_any_finite, np.min(np.where(is_values_finite, values, np.inf), axis=0), 0.0),
        np.where(is_any_finite, np.max(np.where(is_values_finite, values, -np.inf), axis=0), 0.0),
    )


class Matern52Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, squared_distance: torch.Tensor) -> torch.Tensor:
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        (deriv,) = ctx.saved_tensors
        return deriv * grad


class KernelParamsTensor:
    def __init__(self, raw_kernel_params: torch.Tensor) -> None:
        self._raw_kernel_params = raw_kernel_params

    def clone(self) -> torch.Tensor:
        cloned_kernel_params = self._raw_kernel_params.detach()
        cloned_kernel_params.grad = None
        return cloned_kernel_params

    @property
    def inverse_squared_lengthscales(self) -> torch.Tensor:
        return self._raw_kernel_params[:-2]

    @property
    def kernel_scale(self) -> torch.Tensor:
        return self._raw_kernel_params[-2]

    @property
    def noise_var(self) -> torch.Tensor:
        return self._raw_kernel_params[-1]

    @classmethod
    def from_raw_params(
        cls, raw_params: torch.Tensor, minimum_noise: float, deterministic_objective: bool
    ) -> "KernelParamsTensor":
        min_noise_t = torch.tensor([minimum_noise], dtype=torch.float64)
        noise_var = min_noise_t if deterministic_objective else raw_params[-1].exp() + min_noise_t
        return cls(torch.concat([raw_params[:-1].exp(), noise_var]))

    def to_raw_params(self, minimum_noise: float) -> np.ndarray:
        exp_raw_params = self._raw_kernel_params.detach().numpy()
        exp_raw_params[-1] -= 0.99 * minimum_noise
        return np.log(exp_raw_params)


class GPRegressor:
    def __init__(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        kernel_params: torch.Tensor | None = None,
    ) -> None:
        self.kernel_params = KernelParamsTensor(
            raw_kernel_params=(
                torch.ones(X_train.shape[1] + 2, dtype=torch.float64)
                if kernel_params is None
                else kernel_params
            )
        )
        self._X_train = X_train
        self._y_train = y_train
        self._cov_Y_Y_inv: torch.Tensor | None = None
        self._cov_Y_Y_inv_Y: torch.Tensor | None = None

    def _update_kernel_params(self, kernel_params: KernelParamsTensor) -> None:
        self.kernel_params = kernel_params

    def _cache_matrix(self) -> None:
        with torch.no_grad():
            cov_Y_Y = self._kernel(self._X_train, self._X_train).detach().numpy()

        cov_Y_Y[np.diag_indices(self._X_train.shape[0])] += self.kernel_params.noise_var.item()
        cov_Y_Y_inv = np.linalg.inv(cov_Y_Y)
        cov_Y_Y_inv_Y = cov_Y_Y_inv @ self._y_train.numpy()
        self._cov_Y_Y_inv = torch.from_numpy(cov_Y_Y_inv)
        self._cov_Y_Y_inv_Y = torch.from_numpy(cov_Y_Y_inv_Y)

    @property
    def length_scales(self) -> np.ndarray:
        return 1.0 / np.sqrt(self.kernel_params.inverse_squared_lengthscales.detach().numpy())

    def _kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        d2 = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2
        d2 = (d2 * self.kernel_params.inverse_squared_lengthscales).sum(dim=-1)
        return Matern52Kernel.apply(d2) * self.kernel_params.kernel_scale  # type: ignore

    def posterior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._cov_Y_Y_inv_Y is not None and self._cov_Y_Y_inv is not None
        cov_fx_fX = self._kernel(x[..., None, :], self._X_train)[..., 0, :]
        cov_fx_fx = self.kernel_params.kernel_scale
        mean = cov_fx_fX @ self._cov_Y_Y_inv_Y
        var = cov_fx_fx - (cov_fx_fX * (cov_fx_fX @ self._cov_Y_Y_inv)).sum(dim=-1)
        return mean, torch.clamp(var, min=0.0)

    def _marginal_log_likelihood(self) -> torch.Tensor:
        cov_fX_fX = self._kernel(self._X_train, self._X_train)
        n_points = self._X_train.shape[0]
        cov_Y_Y_chol = torch.linalg.cholesky(
            cov_fX_fX + self.kernel_params.noise_var * torch.eye(n_points, dtype=torch.float64)
        )
        logdet = 2 * torch.log(torch.diag(cov_Y_Y_chol)).sum()
        cov_Y_Y_chol_inv_Y = torch.linalg.solve_triangular(
            cov_Y_Y_chol, self._y_train[:, None], upper=False
        )[:, 0]
        return -0.5 * (
            logdet + n_points * math.log(2 * math.pi) + (cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y)
        )

    def _fit_kernel_params(
        self,
        log_prior: Callable[[KernelParamsTensor], torch.Tensor],
        minimum_noise: float,
        deterministic_objective: bool,
        gtol: float,
    ) -> None:
        def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
            raw_params_tensor = torch.from_numpy(raw_params).requires_grad_(True)
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                kernel_params = KernelParamsTensor.from_raw_params(
                    raw_params_tensor, minimum_noise, deterministic_objective
                )
                self._update_kernel_params(kernel_params)
                loss = -self._marginal_log_likelihood() - log_prior(self.kernel_params)
                loss.backward()  # type: ignore
                raw_noise_var_grad = raw_params_tensor.grad[-1]  # type: ignore
                assert not deterministic_objective or raw_noise_var_grad == 0
            return loss.item(), raw_params_tensor.grad.detach().numpy()  # type: ignore

        res = so.minimize(
            loss_func,
            x0=self.kernel_params.to_raw_params(minimum_noise),
            jac=True,
            method="l-bfgs-b",
            options={"gtol": gtol},
        )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        kernel_params_opt = KernelParamsTensor.from_raw_params(
            torch.from_numpy(res.x), minimum_noise, deterministic_objective
        )
        self._update_kernel_params(kernel_params_opt)
        self._cache_matrix()

    def fit_kernel_params(
        self,
        log_prior: Callable[[KernelParamsTensor], torch.Tensor],
        minimum_noise: float,
        deterministic_objective: bool,
        gtol: float = 1e-2,
    ) -> "GPRegressor":
        default_kernel_params = torch.ones(self._X_train.shape[1] + 2, dtype=torch.float64)
        for _ in range(2):
            try:
                self._fit_kernel_params(log_prior, minimum_noise, deterministic_objective, gtol)
                return self
            except RuntimeError:
                self._update_kernel_params(KernelParamsTensor(default_kernel_params.clone()))

        self._update_kernel_params(KernelParamsTensor(default_kernel_params.clone()))
        self._cache_matrix()
        return self
