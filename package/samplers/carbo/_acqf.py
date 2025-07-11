from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import numpy as np
import torch

from ._gp import GPRegressor


class BaseAcquisitionFunc(ABC):
    def __init__(self, length_scales: np.ndarray) -> None:
        self.length_scales = length_scales

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


class UCB(BaseAcquisitionFunc):
    def __init__(self, gpr: GPRegressor, beta: float) -> None:
        self._gpr = gpr
        self._beta = beta
        super().__init__(gpr.length_scales)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        return mean + torch.sqrt(self._beta * var)


class LCB(BaseAcquisitionFunc):
    def __init__(self, gpr: GPRegressor, beta: float) -> None:
        self._gpr = gpr
        self._beta = beta
        super().__init__(gpr.length_scales)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        return mean - torch.sqrt(self._beta * var)


class CombinedUCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        constraints_gpr_list: list[GPRegressor] | None,
        constraints_threshold_list: list[float] | None,
        rho: float,
        beta: float,
    ) -> None:
        self._objective_acqf = UCB(gpr, beta)
        self._constraints_threshold_list = constraints_threshold_list
        self._constraints_acqf_list = (
            [UCB(_gpr, beta) for _gpr in constraints_gpr_list]
            if constraints_gpr_list is not None
            else None
        )
        self._rho = rho
        super().__init__(gpr.length_scales)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        fvals = self._objective_acqf.eval_acqf(x)
        if self._constraints_acqf_list is None:
            return fvals

        assert self._constraints_threshold_list is not None
        _zero = torch.zeros(x.shape[:-1], dtype=torch.float64)
        for threshold, constraint_acqf in zip(
            self._constraints_threshold_list, self._constraints_acqf_list
        ):
            # c >= threshold means to be feasible. constraint_acqf.eval_acqf(x) - threshold is
            # upper confidence bound of the violation amount.
            fvals -= self._rho * torch.maximum(_zero, threshold - constraint_acqf.eval_acqf(x))

        return fvals


class CombinedLCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        constraints_gpr_list: list[GPRegressor] | None,
        constraints_threshold_list: list[float] | None,
        rho: float,
        beta: float,
    ) -> None:
        self._objective_acqf = LCB(gpr, beta)
        self._constraints_threshold_list = constraints_threshold_list
        self._constraints_acqf_list = (
            [LCB(_gpr, beta) for _gpr in constraints_gpr_list]
            if constraints_gpr_list is not None
            else None
        )
        self._rho = rho
        super().__init__(gpr.length_scales)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        fvals = self._objective_acqf.eval_acqf(x)
        if self._constraints_acqf_list is None:
            return fvals

        assert self._constraints_threshold_list is not None
        _zero = torch.zeros(x.shape[:-1], dtype=torch.float64)
        for threshold, constraint_acqf in zip(
            self._constraints_threshold_list, self._constraints_acqf_list
        ):
            # c >= threshold means to be feasible. constraint_acqf.eval_acqf(x) - threshold is
            # lower confidence bound of the violation amount.
            fvals -= self._rho * torch.maximum(_zero, threshold - constraint_acqf.eval_acqf(x))

        return fvals
