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


def _sample_input_noise(
    n_input_noise_samples: int,
    uniform_input_noise_ranges: torch.Tensor | None,
    normal_input_noise_stdevs: torch.Tensor | None,
    seed: int | None,
) -> torch.Tensor:
    def _sample_input_noise_(noise_params: torch.Tensor, gen: SobolGenerator) -> torch.Tensor:
        dim = noise_params.size(0)
        noisy_inds = torch.where(noise_params != 0.0)
        input_noise = torch.zeros(size=(n_input_noise_samples, dim), dtype=torch.float64)
        input_noise[:, noisy_inds[0]] = (
            gen(noisy_inds[0].size(0), n_input_noise_samples, seed) * noise_params[noisy_inds]
        )
        return input_noise

    assert uniform_input_noise_ranges is not None or normal_input_noise_stdevs is not None
    if normal_input_noise_stdevs is not None:
        dim = normal_input_noise_stdevs.size(0)
        noisy_inds = torch.where(normal_input_noise_stdevs != 0.0)
        input_noise = torch.zeros(size=(n_input_noise_samples, dim), dtype=torch.float64)
        input_noise[:, noisy_inds[0]] = (
            _sample_from_normal_sobol(noisy_inds[0].size(0), n_input_noise_samples, seed)
            * normal_input_noise_stdevs[noisy_inds]
        )
        return input_noise
    elif uniform_input_noise_ranges is not None:
        dim = uniform_input_noise_ranges.size(0)
        noisy_inds = torch.where(uniform_input_noise_ranges != 0.0)
        input_noise = torch.zeros(size=(n_input_noise_samples, dim), dtype=torch.float64)
        input_noise[:, noisy_inds[0]] = (
            _sample_from_sobol(noisy_inds[0].size(0), n_input_noise_samples, seed)
            * 2
            * uniform_input_noise_ranges[noisy_inds]
            - uniform_input_noise_ranges[noisy_inds]
        )
        return input_noise
    else:
        raise ValueError(
            "Either `uniform_input_noise_ranges` or `normal_input_noise_stdevs` "
            "must be provided."
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


class LogProbabilityAtRisk(BaseAcquisitionFunc):
    """The logarithm of the probability measure at risk

    When we replace f(x) in VaR with 1[f(x) <= f*], the optimization of the new VaR corresponds to
    that of the mean probability of x with input perturbation being feasible.
    """

    def __init__(
        self,
        gpr_list: list[GPRegressor],
        search_space: SearchSpace,
        threshold_list: list[float],
        n_input_noise_samples: int,
        qmc_seed: int | None,
        uniform_input_noise_ranges: torch.Tensor | None = None,
        normal_input_noise_stdevs: torch.Tensor | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._gpr_list = gpr_list
        self._threshold_list = threshold_list
        rng = np.random.RandomState(qmc_seed)
        self._input_noise = _sample_input_noise(
            n_input_noise_samples,
            uniform_input_noise_ranges,
            normal_input_noise_stdevs,
            seed=rng.random_integers(0, 2**31 - 1, size=1).item(),
        )
        self._stabilizing_noise = stabilizing_noise
        super().__init__(
            length_scales=np.mean([gpr.length_scales for gpr in gpr_list], axis=0),
            search_space=search_space,
        )

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        x_noisy = x.unsqueeze(-2) + self._input_noise
        log_feas_probs = torch.zeros(x_noisy.shape[:-1], dtype=torch.float64)
        for gpr, threshold in zip(self._gpr_list, self._threshold_list):
            means, vars_ = gpr.posterior(x_noisy)
            sigmas = torch.sqrt(vars_ + self._stabilizing_noise)
            # NOTE(nabenabe): integral from a to b of f(x) is integral from -b to -a of f(-x).
            log_feas_probs += torch.special.log_ndtr((means - threshold) / sigmas)
        n_input_noise_samples = len(self._input_noise)
        n_risky_samples = math.ceil(0.05 * n_input_noise_samples)
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
        alpha: float,
        n_input_noise_samples: int,
        n_qmc_samples: int,
        qmc_seed: int | None,
        acqf_type: str,
        uniform_input_noise_ranges: torch.Tensor | None = None,
        normal_input_noise_stdevs: torch.Tensor | None = None,
    ) -> None:
        assert 0 <= alpha <= 1
        self._gpr = gpr
        self._alpha = alpha
        rng = np.random.RandomState(qmc_seed)
        self._input_noise = _sample_input_noise(
            n_input_noise_samples,
            uniform_input_noise_ranges,
            normal_input_noise_stdevs,
            seed=rng.random_integers(0, 2**31 - 1, size=1).item(),
        )
        self._fixed_samples = _sample_from_normal_sobol(
            dim=n_input_noise_samples,
            n_samples=n_qmc_samples,
            seed=rng.random_integers(0, 2**31 - 1, size=1).item(),
        )
        self._acqf_type = acqf_type
        super().__init__(length_scales=gpr.length_scales, search_space=search_space)

    def _value_at_risk(self, x: torch.Tensor) -> torch.Tensor:
        means, covar = self._gpr.joint_posterior(x.unsqueeze(-2) + self._input_noise)
        # TODO: Think of a better way to avoid numerical issue in the Cholesky decomposition.
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
        if self._acqf_type == "mean":
            return self._value_at_risk(x).mean(dim=-1)
        elif self._acqf_type == "nei":
            raise NotImplementedError("NEI is not implemented yet.")
        else:
            raise ValueError(f"Unknown acqf_type: {self._acqf_type}")


class ConstrainedLogValueAtRisk(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        constraints_gpr_list: list[GPRegressor],
        constraints_threshold_list: list[float],
        alpha: float,
        n_input_noise_samples: int,
        n_qmc_samples: int,
        qmc_seed: int | None,
        acqf_type: str,
        uniform_input_noise_ranges: torch.Tensor | None = None,
        normal_input_noise_stdevs: torch.Tensor | None = None,
        stabilizing_noise: float = 1e-12,
    ) -> None:
        self._value_at_risk = ValueAtRisk(
            gpr=gpr,
            search_space=search_space,
            alpha=alpha,
            n_input_noise_samples=n_input_noise_samples,
            n_qmc_samples=n_qmc_samples,
            qmc_seed=qmc_seed,
            acqf_type=acqf_type,
            uniform_input_noise_ranges=uniform_input_noise_ranges,
            normal_input_noise_stdevs=normal_input_noise_stdevs,
        )
        self._log_prob_at_risk = LogProbabilityAtRisk(
            gpr_list=constraints_gpr_list,
            search_space=search_space,
            threshold_list=constraints_threshold_list,
            n_input_noise_samples=n_input_noise_samples,
            qmc_seed=qmc_seed,
            uniform_input_noise_ranges=uniform_input_noise_ranges,
            normal_input_noise_stdevs=normal_input_noise_stdevs,
            stabilizing_noise=stabilizing_noise,
        )
        super().__init__(self._value_at_risk.length_scales, search_space=search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        return self._value_at_risk.eval_acqf(x).clamp_min_(
            _EPS
        ).log_() + self._log_prob_at_risk.eval_acqf(x)
