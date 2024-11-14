from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.logging import get_logger
from optuna.samplers import TPESampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.sampler import _split_trials
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from .components import GammaFunc
from .components import WeightFunc
from .parzen_estimator import _CustomizableParzenEstimator
from .parzen_estimator import _CustomizableParzenEstimatorParameters


_logger = get_logger(f"optuna.{__name__}")


def _ctpe_split_trials(
    study: Study, trials: list[FrozenTrial], n_below: int, enable_constriants: bool
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    return [], []


class cTPESampler(TPESampler):
    def __init__(
        self,
        *,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        seed: int | None = None,
        categorical_prior_weight: float | None = 0.2,
        multivariate: bool = True,
        b_magic_exponent: float = np.inf,
        min_bandwidth_factor: float = 0.01,
        gamma_strategy: str = "sqrt",
        gamma_beta: float = 0.25,
        weight_strategy: str = "uniform",
        bandwidth_strategy: str = "scott",
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ):
        gamma = GammaFunc(strategy=gamma_strategy, beta=gamma_beta)
        weights = WeightFunc(strategy=weight_strategy)
        super().__init__(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=True,
            warn_independent_sampling=False,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            weights=weights,
            seed=seed,
            multivariate=multivariate,
            constraints_func=constraints_func,
        )
        self._parzen_estimator_cls = _CustomizableParzenEstimator
        self._parzen_estimator_parameters = _CustomizableParzenEstimatorParameters(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            weights=weights,
            multivariate=multivariate,
            b_magic_exponent=b_magic_exponent,
            min_bandwidth_factor=min_bandwidth_factor,
            bandwidth_strategy=bandwidth_strategy,
            categorical_prior_weight=categorical_prior_weight,
        )

    def _sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if study._is_multi_objective():
            _logger.warning("Multi-objective c-TPE does not exist in the original paper.")

        trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
        # n_below_feasible = self._gamma(len(trials))
        # constraints_vals = np.asarray([self._constraints_func(t) for t in trials])
        n_below = ...
        # qs = ...

        # We divide data into below and above.
        # n_trials = len(trials)
        below_trials, above_trials = _split_trials(
            study, trials, n_below, constraints_enabled=False
        )
        mpe_below = self._build_parzen_estimator(
            study, search_space, below_trials, handle_below=True
        )
        mpe_above = self._build_parzen_estimator(
            study, search_space, above_trials, handle_below=False
        )

        samples_below = mpe_below.sample(self._rng.rng, self._n_ei_candidates)
        acq_func_vals = self._compute_acquisition_func(samples_below, mpe_below, mpe_above, [])
        ret = TPESampler._compare(samples_below, acq_func_vals)

        for param_name, dist in search_space.items():
            ret[param_name] = dist.to_external_repr(ret[param_name])

        return ret

    def _compute_acquisition_func(
        self,
        samples: dict[str, np.ndarray],
        mpes_below: list[_ParzenEstimator],
        mpes_above: list[_ParzenEstimator],
        quantiles: list[float],
    ) -> np.ndarray:
        _EPS = 1e-12
        assert len(mpes_above) == len(mpes_below) == len(quantiles)
        lls_above = np.asarray([mpe_above.log_pdf(samples) for mpe_above in mpes_above])
        lls_below = np.asarray([mpe_below.log_pdf(samples) for mpe_below in mpes_below])
        _q = np.asarray(quantiles)[:, np.newaxis]
        log_first_term = np.log(_q + _EPS)
        log_second_term = np.log(1.0 - _q + _EPS) + lls_above - lls_below
        acq_func_vals = np.sum(-np.logaddexp(log_first_term, log_second_term), axis=0)
        return acq_func_vals
