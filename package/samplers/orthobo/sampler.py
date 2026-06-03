from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any
from typing import List
from typing import Tuple

from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from optuna_integration import BoTorchSampler
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal


# ---------------------------------------------------------------------------
# Hyperparam extraction utils
# ---------------------------------------------------------------------------


@dataclass
class ThetaField:
    path: str
    shape: torch.Size
    size: int
    transform: str = "log"


@dataclass
class ThetaSpec:
    fields: List[ThetaField]

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.fields)


_SUPPORTED_POSITIVE_PATHS = [
    "likelihood.noise",
    "covar_module.outputscale",
    "covar_module.variance",
    "covar_module.lengthscale",
]


def _get_attr(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set_attr(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _discover_theta_spec(model: Any) -> ThetaSpec:
    fields = []
    for path in _SUPPORTED_POSITIVE_PATHS:
        try:
            val = _get_attr(model, path)
            if torch.is_tensor(val):
                fields.append(ThetaField(path=path, shape=val.shape, size=val.numel()))
        except AttributeError:
            continue
    if not fields:
        raise RuntimeError("No supported GP hyperparameters found.")
    return ThetaSpec(fields=fields)


def _extract_log_theta(model: Any) -> Tuple[Tensor, ThetaSpec]:
    spec = _discover_theta_spec(model)
    pieces = [
        _get_attr(model, f.path).detach().reshape(-1).clamp_min(1e-12).log() for f in spec.fields
    ]
    return torch.cat(pieces), spec


def _set_theta_from_log(model: Any, u: Tensor, spec: ThetaSpec) -> None:
    offset = 0
    with torch.no_grad():
        for field in spec.fields:
            chunk = u[offset : offset + field.size].view(field.shape).exp()
            _set_attr(model, field.path, chunk)
            offset += field.size
    model.prediction_strategy = None


# ---------------------------------------------------------------------------
# Hyperposterior (diagonal Laplace approximation)
# ---------------------------------------------------------------------------


@dataclass
class GaussianHyperPosterior:
    mean: Tensor
    cov: Tensor

    def sample(self, S: int) -> Tensor:
        return MultivariateNormal(self.mean, covariance_matrix=self.cov).rsample((S,))

    def score(self, u: Tensor) -> Tensor:
        diff = u - self.mean
        return -torch.linalg.solve(self.cov, diff.T).T


def _neg_log_post(u: Tensor, model_template: Any, spec: ThetaSpec) -> Tensor:
    model = copy.deepcopy(model_template)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    model.likelihood.train()
    _set_theta_from_log(model, u, spec)
    output = model(model.train_inputs[0])
    return -mll(output, model.train_targets)


def _build_hyperposterior(
    fitted_model: Any,
    hess_jitter: float = 1e-4,
    max_log_std: float = 0.10,
) -> Tuple[GaussianHyperPosterior, ThetaSpec]:
    u_hat, spec = _extract_log_theta(fitted_model)
    u_hat = u_hat.detach().clone().requires_grad_(True)

    H = torch.autograd.functional.hessian(
        lambda u: _neg_log_post(u, fitted_model, spec), u_hat
    ).detach()
    H = 0.5 * (H + H.T)
    H_diag = torch.diag(H).clamp_min(hess_jitter)
    std = torch.clamp(1.0 / torch.sqrt(H_diag), max=max_log_std)

    posterior = GaussianHyperPosterior(
        mean=u_hat.detach(),
        cov=torch.diag(std.pow(2)).detach(),
    )
    return posterior, spec


def _sample_valid_thetas(
    model: Any,
    hyperposterior: GaussianHyperPosterior,
    spec: ThetaSpec,
    S: int,
    train_x: Tensor,
    max_tries: int = 1000,
) -> Tuple[Tensor, List[Any]]:
    accepted_models: list[Any] = []
    accepted_thetas: list[Tensor] = []
    thetas = hyperposterior.sample(2 * S)
    idx = tries = 0

    while len(accepted_models) < S and tries < max_tries:
        theta = thetas[idx]
        tries += 1
        try:
            m = copy.deepcopy(model)
            m.prediction_strategy = None
            _set_theta_from_log(m, theta, spec)
            m.posterior(train_x[:1])  # validate
            accepted_models.append(m)
            accepted_thetas.append(theta)
        except Exception:
            pass
        idx += 1

    if len(accepted_models) < S:
        raise RuntimeError(
            f"Could not obtain {S} valid theta samples after {max_tries} tries. "
            "Consider increasing hess_jitter or reducing mc_budget."
        )
    return torch.stack(accepted_thetas, dim=0), accepted_models


# ---------------------------------------------------------------------------
# Orthogonal acquisition function
# ---------------------------------------------------------------------------


class _OrthogonalLogEI(AcquisitionFunction):
    def __init__(
        self,
        model: Any,
        best_f: Tensor,
        hyperposterior: GaussianHyperPosterior,
        theta_samples: Tensor,
        theta_models: List[Any],
        use_orthogonal_correction: bool = True,
        eps: float = 1e-12,
        cov_jitter: float = 1e-6,
    ) -> None:
        super().__init__(model=model)
        self.eps = eps
        self.cov_jitter = cov_jitter
        self.use_orthogonal_correction = use_orthogonal_correction

        g = hyperposterior.score(theta_samples)
        self.g = g
        self.g_centered = g - g.mean(dim=0, keepdim=True)
        self.g_cov = torch.cov(g.T)

        self.theta_acqfns = [LogExpectedImprovement(model=m, best_f=best_f) for m in theta_models]

    def forward(self, X: Tensor) -> Tensor:
        h = torch.stack([torch.exp(fn(X)) for fn in self.theta_acqfns], dim=0)

        if not self.use_orthogonal_correction:
            return torch.log(torch.clamp_min(h.mean(dim=0), self.eps))

        S = self.g.shape[0]
        h_centered = h - h.mean(dim=0, keepdim=True)
        cov_gg = self.g_cov + self.cov_jitter * torch.eye(
            self.g_cov.shape[0], dtype=self.g_cov.dtype, device=self.g_cov.device
        )
        cov_gh = (self.g_centered.T @ h_centered) / max(S - 1, 1)
        gamma = torch.linalg.solve(cov_gg, cov_gh)
        orth = h - (self.g @ gamma)
        return torch.log(torch.clamp_min(orth.mean(dim=0), self.eps))


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class OrthoBoSampler(BoTorchSampler):
    """
    Orthogonalized Bayesian Optimization sampler for Optuna.

    Args:
        n_startup_trials:
            Number of initial quasi-random (Sobol) trials before BO begins.
            Defaults to 10.
        mc_budget:
            Number of GP models sampled from the hyperposterior (S in paper).
            Higher values reduce variance but increase compute. Defaults to 64.
        use_orthogonal_correction:
            If True (default), applies the orthogonal score-function control
            variate for variance reduction (OrthoBO mode).
            If False, falls back to plain MC marginalisation over the
            hyperposterior (Naive Marginal BO mode).
        seed:
            Random seed for the Sobol startup sampler. Set for reproducibility.
            Defaults to None.
    """

    def __init__(
        self,
        n_startup_trials: int = 10,
        mc_budget: int = 64,
        use_orthogonal_correction: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            candidates_func=self._get_candidates,
            n_startup_trials=n_startup_trials,
            seed=seed,
        )
        self.mc_budget = mc_budget
        self.use_orthogonal_correction = use_orthogonal_correction

    def _get_candidates(
        self,
        train_x: Tensor,
        train_obj: Tensor,
        train_con: Tensor | None,
        bounds: Tensor,
        pending_x: Tensor | None,
    ) -> Tensor:
        dim = train_x.shape[-1]

        model = SingleTaskGP(
            train_x,
            train_obj,
            input_transform=Normalize(d=dim, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        hyperposterior, spec = _build_hyperposterior(model)
        theta_samples, theta_models = _sample_valid_thetas(
            model=model,
            hyperposterior=hyperposterior,
            spec=spec,
            S=self.mc_budget,
            train_x=train_x,
        )

        acqf = _OrthogonalLogEI(
            model=model,
            best_f=train_obj.max(),
            hyperposterior=hyperposterior,
            theta_samples=theta_samples,
            theta_models=theta_models,
            use_orthogonal_correction=self.use_orthogonal_correction,
        )

        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=256,
        )

        return candidate.clamp(bounds[0], bounds[1])
