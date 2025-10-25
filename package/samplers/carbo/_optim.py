from __future__ import annotations

import math
import threading

import numpy as np
import scipy.optimize as so
import scipy.stats.qmc as qmc

from ._acqf import BaseAcquisitionFunc
from ._acqf import CombinedLCB
from ._acqf import CombinedUCB
from ._gp import GPRegressor
from ._scipy_blas_thread_patch import single_blas_thread_if_scipy_v1_15_or_newer


_threading_lock = threading.Lock()


def sample_normalized_params(n: int, dim: int, rng: np.random.RandomState | None) -> np.ndarray:
    rng = rng or np.random.RandomState()
    with _threading_lock:
        qmc_engine = qmc.Sobol(dim, scramble=True, seed=rng.randint(np.iinfo(np.int32).max))
    return qmc_engine.random(n)


def _gradient_descent(
    acqf: BaseAcquisitionFunc, initial_params: np.ndarray, bounds: np.ndarray, *, tol: float = 1e-4
) -> tuple[np.ndarray, float]:
    lengthscales = acqf.length_scales
    normalized_params = initial_params.copy()
    initial_fval = float(acqf.eval_acqf_no_grad(initial_params.copy()))

    def acqf_with_grad(scaled_x: np.ndarray) -> tuple[float, np.ndarray]:
        normalized_params = scaled_x * lengthscales
        (fval, grad) = acqf.eval_acqf_with_grad(normalized_params)
        return fval, grad * lengthscales

    with single_blas_thread_if_scipy_v1_15_or_newer():
        scaled_x_opt, fval_opt, info = so.fmin_l_bfgs_b(
            func=acqf_with_grad,
            x0=normalized_params / lengthscales,
            bounds=bounds / lengthscales[:, np.newaxis],
            pgtol=math.sqrt(tol),
            maxiter=200,
        )

    if fval_opt < initial_fval and info["nit"] > 0:  # Improved.
        normalized_params = scaled_x_opt * lengthscales
        return normalized_params, fval_opt

    return initial_params, initial_fval  # No improvement.


def _create_bounds(x_local: np.ndarray, local_radius: float) -> np.ndarray:
    bounds = np.empty((len(x_local), 2), dtype=float)
    bounds[:, 0] = np.maximum(0.0, x_local - local_radius)
    bounds[:, 1] = np.minimum(1.0, x_local + local_radius)
    return bounds


def suggest_by_carbo(
    *,
    gpr: GPRegressor,
    constraints_gpr_list: list[GPRegressor] | None,
    constraints_threshold_list: list[float] | None,
    rng: np.random.RandomState | None,
    rho: float,
    beta: float,
    n_local_search: int,
    local_radius: float,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    dim = len(gpr.length_scales)
    local_params = sample_normalized_params(n_local_search, dim, rng=rng)
    robust_x_local: np.ndarray | None = None
    robust_f_local = -np.inf
    ucb_acqf = CombinedUCB(
        gpr=gpr,
        constraints_gpr_list=constraints_gpr_list,
        constraints_threshold_list=constraints_threshold_list,
        rho=rho,
        beta=beta,
    )
    for x_local in local_params:
        bounds = _create_bounds(x_local, local_radius)
        _, f = _gradient_descent(ucb_acqf, x_local, bounds, tol=tol)
        if f > robust_f_local:
            robust_x_local = x_local.copy()
            robust_f_local = f

    lcb_acqf = CombinedLCB(
        gpr=gpr,
        constraints_gpr_list=constraints_gpr_list,
        constraints_threshold_list=constraints_threshold_list,
        rho=rho,
        beta=beta,
    )
    assert isinstance(robust_x_local, np.ndarray)
    bounds = _create_bounds(robust_x_local, local_radius)
    return robust_x_local, *_gradient_descent(lcb_acqf, robust_x_local, bounds, tol=tol)


def evaluate_by_carbo(
    *,
    robust_params: np.ndarray,
    gpr: GPRegressor,
    constraints_gpr_list: list[GPRegressor] | None,
    constraints_threshold_list: list[float] | None,
    rng: np.random.RandomState | None,
    rho: float,
    beta: float,
    n_local_search: int,
    local_radius: float,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float]:
    lcb_acqf = CombinedLCB(
        gpr=gpr,
        constraints_gpr_list=constraints_gpr_list,
        constraints_threshold_list=constraints_threshold_list,
        rho=rho,
        beta=beta,
    )
    bounds = _create_bounds(robust_params, local_radius)
    return _gradient_descent(lcb_acqf, robust_params, bounds, tol=tol)
