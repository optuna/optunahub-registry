"""Level-set estimation for the DRPTR measure over a finite design x environment grid.

Implements sections 2 and 3.1 of Inatsu, Iwazaki & Takeuchi, "Active Learning for
Distributionally Robust Level-Set Estimation", ICML 2021.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve

from ._drptr import drptr_binary


def gaussian_kernel(
    x_points: np.ndarray, w_points: np.ndarray, sigma_f: float = 1.0, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """The paper's kernel, ``sigma_f^2 exp(-{(x-x')^2 + (w-w')^2} / scale)``.

    Note that ``scale`` divides the squared distance directly, as printed in section 5 of
    the paper. It is therefore the square of a length scale, not a length scale: an RBF
    with length scale ``l`` corresponds to ``scale = 2 * l**2``.

    Grid points are ordered with the design variable outermost, so the flat index of
    ``(i, j)`` is ``i * len(w_points) + j``.

    Args:
        x_points: One-dimensional array of design points.
        w_points: One-dimensional array of environmental points.
        sigma_f: Prior standard deviation of the process.
        scale: Divisor of the squared distance. Must be positive.

    Returns:
        The kernel matrix over the joint grid, and the joint grid points themselves.
    """
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}.")
    if x_points.ndim != 1 or w_points.ndim != 1:
        raise ValueError("x_points and w_points must be one-dimensional.")
    points = np.stack(np.meshgrid(x_points, w_points, indexing="ij"), axis=-1).reshape(-1, 2)
    sq_dist = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    return sigma_f**2 * np.exp(-sq_dist / scale), points


class DRLevelSetEstimator:
    """GP posterior on a finite joint grid, and the DRPTR classification it induces.

    Args:
        kernel: Kernel matrix over the ``n_x * n_w`` joint grid points. Copied.
        n_x: Number of design points.
        n_w: Number of environmental points.
        p_ref: Reference p.m.f. over the environmental points. Copied.
        eps: Radius of the L1 ambiguity set around ``p_ref``.
        h: Threshold defining the robustness event ``f(x, w) > h``.
        alpha: Level defining the reliable set ``H = {x : F(x) > alpha}``.
        eta: Accuracy parameter of section 3.1. With ``eta = 0`` the classification cannot
            misclassify, given valid credible intervals, but boundary points may stay
            unclassified forever. A positive value buys termination in exchange for a
            bounded misclassification loss; see Theorem 4.1. Note that with ``eta > 0`` the
            lower bound ``l_ind`` brackets ``1[f > h - eta]`` rather than ``1[f > h]``, so
            ``lF`` is no longer a lower bound on ``F`` and ``H`` may contain points whose
            true DRPTR is at or below ``alpha``.
        noise: Observation noise variance. Must be positive.
    """

    def __init__(
        self,
        kernel: np.ndarray,
        n_x: int,
        n_w: int,
        p_ref: np.ndarray,
        eps: float,
        h: float,
        alpha: float,
        eta: float = 0.0,
        noise: float = 1e-4,
    ) -> None:
        p_ref = np.array(p_ref, dtype=float)
        if kernel.shape != (n_x * n_w, n_x * n_w):
            raise ValueError(f"kernel must be ({n_x * n_w}, {n_x * n_w}), got {kernel.shape}.")
        if p_ref.shape != (n_w,):
            raise ValueError(f"p_ref must have shape ({n_w},), got {p_ref.shape}.")
        if not np.isclose(p_ref.sum(), 1.0) or (p_ref < 0).any():
            raise ValueError("p_ref must be a probability mass function.")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}.")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}.")
        if eta < 0.0:
            raise ValueError(f"eta must be non-negative, got {eta}.")
        if noise <= 0.0:
            raise ValueError(f"noise must be positive, got {noise}.")

        self._kernel = np.array(kernel, dtype=float)
        self._prior_var = np.diag(self._kernel).copy()
        self.n_x = n_x
        self.n_w = n_w
        self.p_ref = p_ref
        self.eps = eps
        self.h = h
        self.alpha = alpha
        self.eta = eta
        self.noise = noise
        self._observed: list[int] = []
        self._values: list[float] = []

    @property
    def kernel(self) -> np.ndarray:
        """The kernel matrix. Treat as read-only."""
        return self._kernel

    @property
    def n_observations(self) -> int:
        return len(self._observed)

    @property
    def observed_indices(self) -> list[int]:
        """Flat grid indices observed so far, in order."""
        return list(self._observed)

    def flat(self, ix: int, iw: int) -> int:
        """Flat index of the joint grid point ``(ix, iw)``."""
        if not 0 <= ix < self.n_x:
            raise IndexError(f"ix must lie in [0, {self.n_x}), got {ix}.")
        if not 0 <= iw < self.n_w:
            raise IndexError(f"iw must lie in [0, {self.n_w}), got {iw}.")
        return ix * self.n_w + iw

    def observe(self, ix: int, iw: int, y: float) -> None:
        """Record an observation of ``f`` at the joint grid point ``(ix, iw)``."""
        self._observed.append(self.flat(ix, iw))
        self._values.append(float(y))

    def reset(self) -> None:
        """Discard every observation."""
        self._observed.clear()
        self._values.clear()

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and marginal variance over every joint grid point."""
        mean, cov = self.posterior_joint()
        return mean, np.diag(cov).copy()

    def posterior_joint(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and full covariance over every joint grid point."""
        if not self._observed:
            return np.zeros(self._kernel.shape[0]), self._kernel.copy()
        obs = np.asarray(self._observed)
        values = np.asarray(self._values, dtype=float)
        gram = self._kernel[np.ix_(obs, obs)] + self.noise * np.eye(len(obs))
        cross = self._kernel[:, obs]
        chol = cho_factor(gram, lower=True)
        mean = cross @ cho_solve(chol, values)
        cov = self._kernel - cross @ cho_solve(chol, cross.T)
        diag = np.diag(cov)
        if diag.min() < -1e-8:
            raise FloatingPointError(
                f"Posterior variance went negative ({diag.min():.3e}); the kernel is likely "
                "ill-conditioned. Increase `noise` or widen the kernel scale."
            )
        np.fill_diagonal(cov, np.clip(diag, 1e-12, None))
        return mean, cov

    def indicator_bounds(
        self, beta: float, posterior: tuple[np.ndarray, np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Credible bounds on the robustness indicator, section 3.1.

        The lower bound brackets ``1[f(x,w) > h - eta]`` from below and the upper bound
        brackets ``1[f(x,w) > h]`` from above, so the pair is only a credible interval for
        ``1[f > h]`` when ``eta == 0``.

        Args:
            beta: Squared width of the credible interval. The interval is
                ``mean +/- sqrt(beta) * sd``, so this is the square of the usual coefficient.
            posterior: An already-computed ``(mean, variance)`` pair, to avoid recomputing.
                The second element must be a variance, not a standard deviation.
        """
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}.")
        mean, var = self.posterior() if posterior is None else posterior
        sd = np.sqrt(var)
        lower, upper = mean - np.sqrt(beta) * sd, mean + np.sqrt(beta) * sd
        certainly_above = lower > self.h - self.eta
        possibly_above = upper > self.h
        l_ind = certainly_above.astype(float)
        u_ind = (certainly_above | possibly_above).astype(float)
        return l_ind.reshape(self.n_x, self.n_w), u_ind.reshape(self.n_x, self.n_w)

    def drptr_bounds(
        self, beta: float, posterior: tuple[np.ndarray, np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``l^F`` and ``u^F`` of eq. (3.1). Both are infima over the ambiguity set."""
        l_ind, u_ind = self.indicator_bounds(beta, posterior)
        lower = np.array(
            [
                drptr_binary(l_ind[i], self.p_ref, self.eps, l_ind[i] @ self.p_ref)
                for i in range(self.n_x)
            ]
        )
        upper = np.array(
            [
                drptr_binary(u_ind[i], self.p_ref, self.eps, u_ind[i] @ self.p_ref)
                for i in range(self.n_x)
            ]
        )
        return lower, upper

    def classify(
        self, beta: float, posterior: tuple[np.ndarray, np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boolean masks ``(H_t, L_t, U_t)`` over the ``n_x`` design points."""
        lower, upper = self.drptr_bounds(beta, posterior)
        reliable = lower > self.alpha
        unreliable = upper <= self.alpha
        return reliable, unreliable, ~(reliable | unreliable)


def true_drptr(f_grid: np.ndarray, p_ref: np.ndarray, eps: float, h: float) -> np.ndarray:
    """Ground-truth ``F(x)`` for a fully known ``f`` on the grid, for evaluation."""
    p_ref = np.asarray(p_ref, dtype=float)
    ind = (f_grid > h).astype(float)
    return np.array(
        [drptr_binary(ind[i], p_ref, eps, ind[i] @ p_ref) for i in range(f_grid.shape[0])]
    )
