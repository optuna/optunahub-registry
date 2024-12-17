from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.logging import get_logger
from optuna.samplers import TPESampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from .components import GammaFunc
from .components import WeightFunc
from .parzen_estimator import _CustomizableParzenEstimator
from .parzen_estimator import _CustomizableParzenEstimatorParameters


_logger = get_logger(f"optuna.{__name__}")


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
        use_min_bandwidth_discrete: bool = True,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ):
        if constraints_func is None:
            raise ValueError(
                f"{self.__class__.__name__} must take constraints_func, but got None."
            )

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
            use_min_bandwidth_discrete=use_min_bandwidth_discrete,
        )

    def _warning_multi_objective_for_ctpe(self, study: Study) -> None:
        """TODO: Use this routine once c-TPE supports multi-objective optimization.
        if study._is_multi_objective():
            def _get_additional_msg() -> str:
                beta = getattr(self._gamma, "_beta", None)
                strategy = getattr(self._gamma, "_strategy", None)
                if beta != 0.15 or strategy != "linear":
                    return ""

                return (
                    "Note that the original MOTPE uses beta=0.15 and strategy='sqrt', but "
                    f"beta={beta} and strategy='{strategy}' are used in this study."
                )

            _logger.warning(
                "Multi-objective c-TPE does not exist in the original paper, "
                "but sampling will be performed by c-TPE based on Optuna MOTPE. "
                f"{_get_additional_msg()}"
            )
        """
        self._raise_error_if_multi_objective(study)

    def _build_parzen_estimators_for_constraints_and_get_quantiles(
        self,
        trials: list[FrozenTrial],
        study: Study,
        search_space: dict[str, BaseDistribution],
        constraints_vals: np.ndarray,
    ) -> tuple[list[_ParzenEstimator], list[_ParzenEstimator], list[float]]:
        mpes_below: list[_ParzenEstimator] = []
        mpes_above: list[_ParzenEstimator] = []
        quantiles: list[float] = []
        for constraint_vals in constraints_vals.T:
            is_satisfied = (constraint_vals <= 0) | (constraint_vals == min(constraint_vals))
            satisfied_trials = [t for t, include in zip(trials, is_satisfied) if include]
            unsatisfied_trials = [t for t, exclude in zip(trials, is_satisfied) if not exclude]
            mpes_below.append(
                self._build_parzen_estimator(
                    study, search_space, satisfied_trials, handle_below=False
                )
            )
            mpes_above.append(
                self._build_parzen_estimator(
                    study, search_space, unsatisfied_trials, handle_below=False
                )
            )
            quantiles.append(len(satisfied_trials) / max(1, len(trials)))

        return mpes_below, mpes_above, quantiles

    def _sample(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        self._warning_multi_objective_for_ctpe(study)
        trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
        constraints_vals = np.asarray([self._constraints_func(t) for t in trials])
        (mpes_below, mpes_above, quantiles) = (
            self._build_parzen_estimators_for_constraints_and_get_quantiles(
                trials, study, search_space, constraints_vals
            )
        )

        n_below_feasible = self._gamma(len(trials))
        below_trials, above_trials = _split_trials_for_ctpe(
            study, trials, n_below_feasible, is_feasible=np.all(constraints_vals <= 0, axis=-1)
        )
        mpes_below.append(
            self._build_parzen_estimator(study, search_space, below_trials, handle_below=True)
        )
        mpes_above.append(
            self._build_parzen_estimator(study, search_space, above_trials, handle_below=False)
        )
        quantiles.append(len(below_trials) / max(1, len(trials)))

        _samples_below: dict[str, list[np.ndarray]] = {
            param_name: [] for param_name in search_space
        }
        for mpe in mpes_below:
            for param_name, samples in mpe.sample(self._rng.rng, self._n_ei_candidates).items():
                _samples_below[param_name].append(samples)

        samples_below = {
            param_name: np.hstack(samples) for param_name, samples in _samples_below.items()
        }
        acq_func_vals = self._compute_acquisition_func(
            samples_below, mpes_below, mpes_above, quantiles
        )
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


def _split_trials_for_ctpe(
    study: Study, trials: list[FrozenTrial], n_below_feasible: int, is_feasible: np.ndarray
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    if len(trials) == 0:
        return [], []
    if np.count_nonzero(is_feasible) < n_below_feasible or len(trials) == n_below_feasible:
        return trials, []
    if n_below_feasible == 0:
        return [], trials

    loss_vals = np.asarray([t.values for t in trials])
    loss_vals *= np.asarray([1 if d == StudyDirection.MINIMIZE else -1 for d in study.directions])
    if study._is_multi_objective():
        return _split_trials_for_multi_objective_ctpe(loss_vals, n_below_feasible, is_feasible)
    else:
        order = np.argsort(loss_vals[:, 0])
        n_below = np.searchsorted(np.cumsum(is_feasible[order]), n_below_feasible) + 1
        indices_below = set(np.arange(len(trials))[order[:n_below]])
        below_trials = [t for i, t in enumerate(trials) if i in indices_below]
        above_trials = [t for i, t in enumerate(trials) if i not in indices_below]
        return below_trials, above_trials


def _split_trials_for_multi_objective_ctpe(
    loss_vals: np.ndarray, n_below_feasible: int, is_feasible: np.ndarray
) -> tuple[list[FrozenTrial], list[FrozenTrial]]:
    assert 0 < n_below_feasible <= np.count_nonzero(is_feasible)
    assert n_below_feasible < len(loss_vals)
    raise ValueError("c-TPE has not supported multi-objective optimization yet.")
