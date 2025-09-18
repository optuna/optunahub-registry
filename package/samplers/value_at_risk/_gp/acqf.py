from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import math
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from typing import Protocol

    from optuna._gp.gp import GPRegressor
    from optuna._gp.search_space import SearchSpace
    import torch

    class SobolGenerator(Protocol):
        def __call__(self, dim: int, n_samples: int, seed: int | None) -> torch.Tensor:
            raise NotImplementedError

else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


_SQRT_HALF = math.sqrt(0.5)
_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
_SQRT_HALF_PI = math.sqrt(0.5 * math.pi)
_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))
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


class ValueAtRisk(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        alpha: float,
        n_input_noise_samples: int,
        n_qmc_samples: int,
        qmc_seed: int | None,
        uniform_input_noise_ranges: torch.Tensor | None = None,
        normal_input_noise_stdevs: torch.Tensor | None = None,
    ) -> None:
        assert 0 <= alpha <= 1
        self._gpr = gpr
        self._alpha = alpha
        rng = np.random.RandomState(qmc_seed)
        self._input_noise = self._sample_input_noise(
            n_input_noise_samples, uniform_input_noise_ranges, normal_input_noise_stdevs, rng
        )
        seed = rng.random_integers(0, 2**31 - 1, size=1).item()
        self._fixed_samples = _sample_from_normal_sobol(
            dim=n_input_noise_samples, n_samples=n_qmc_samples, seed=seed
        )
        super().__init__(length_scales=gpr.length_scales, search_space=search_space)

    @staticmethod
    def _sample_input_noise(
        n_input_noise_samples: int,
        uniform_input_noise_ranges: torch.Tensor | None,
        normal_input_noise_stdevs: torch.Tensor | None,
        rng: np.random.RandomState,
    ) -> torch.Tensor:
        seed = rng.random_integers(0, 2**31 - 1, size=1).item()

        def _sample_input_noise(noise_params: torch.Tensor, gen: SobolGenerator) -> torch.Tensor:
            dim = noise_params.size(0)
            noisy_inds = torch.where(noise_params != 0.0)
            input_noise = torch.zeros(size=(n_input_noise_samples, dim), dtype=torch.float64)
            input_noise[:, noisy_inds[0]] = (
                gen(noisy_inds[0].size(0), n_input_noise_samples, seed) * noise_params[noisy_inds]
            )
            return input_noise

        assert uniform_input_noise_ranges is not None or normal_input_noise_stdevs is not None
        if normal_input_noise_stdevs is not None:
            return _sample_input_noise(normal_input_noise_stdevs, _sample_from_normal_sobol)
        elif uniform_input_noise_ranges is not None:
            return _sample_input_noise(uniform_input_noise_ranges, _sample_from_sobol)
        else:
            raise ValueError(
                "Either `uniform_input_noise_ranges` or `normal_input_noise_stdevs` "
                "must be provided."
            )

    def _value_at_risk(self, x: torch.Tensor) -> torch.Tensor:
        means, covar = self._gpr.joint_posterior(x.unsqueeze(-2) + self._input_noise)
        L, _ = torch.linalg.cholesky_ex(covar)
        posterior_samples = means.unsqueeze(-2) + self._fixed_samples @ L
        # If CVaR, use torch.topk instead of torch.quantile.
        return torch.quantile(posterior_samples, q=self._alpha, dim=-1)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Adapt to NEI.
        1. Generate posterior samples for each X_train + input_noise. (cache)
        2. Generate posterior samples for each x + input noise. (Use path-wise conditioning)
        3. Use the maximum VaR (cache) for each MC sample as the f0 in NEI. (Denote it as f0[i])
        4. Then compute (mc_value_at_risk - f0).clamp_min(0).mean()
        Appendix B.2 of https://www.robots.ox.ac.uk/~mosb/public/pdf/136/full_thesis.pdf
        """
        return self._value_at_risk(x).mean(dim=-1)
