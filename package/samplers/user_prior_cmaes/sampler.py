from __future__ import annotations

import math
from typing import Any

import cmaes
import numpy as np
from optuna import Study
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import CmaClass
from optuna.samplers import CmaEsSampler
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


class UserPriorCmaEsSampler(CmaEsSampler):
    """A sampler using `cmaes <https://github.com/CyberAgentAILab/cmaes>`__ as the backend with user prior.

    Please check ``CmaEsSampler`` in Optuna for more details of each argument.
    This class modified the arguments ``x0`` and ``sigma0`` in ``CmaEsSampler`` of Optuna.
    Furthermore, due to the incompatibility,
    This class does not support ``source_trials`` and ``use_separable_cma`` due to their incompatibility.

    Args:
        x0:
            A dictionary of an initial parameter values for CMA-ES. By default, the mean of ``low``
            and ``high`` for each distribution is used. Note that ``x0`` is sampled uniformly
            within the search space domain for each restart if you specify ``restart_strategy``
            argument.

        sigma0:
            Initial standard deviation of CMA-ES. By default, ``sigma0`` is set to
            ``min_range / 6``, where ``min_range`` denotes the minimum range of the distributions
            in the search space.
    """  # NOQA: E501

    def __init__(
        self,
        param_names: list[str],
        mu0: np.ndarray,
        cov0: np.ndarray,
        n_startup_trials: int = 1,
        independent_sampler: BaseSampler | None = None,
        warn_independent_sampling: bool = True,
        seed: int | None = None,
        *,
        consider_pruned_trials: bool = False,
        restart_strategy: str | None = None,
        popsize: int | None = None,
        inc_popsize: int = 2,
        with_margin: bool = False,
        lr_adapt: bool = False,
    ) -> None:
        super().__init__(
            x0=None,
            sigma0=None,
            n_startup_trials=n_startup_trials,
            independent_sampler=independent_sampler,
            warn_independent_sampling=warn_independent_sampling,
            seed=seed,
            consider_pruned_trials=consider_pruned_trials,
            restart_strategy=restart_strategy,
            popsize=popsize,
            inc_popsize=inc_popsize,
            use_separable_cma=False,
            with_margin=with_margin,
            lr_adapt=lr_adapt,
            source_trials=None,
        )
        self._validate_user_prior(param_names, mu0, cov0)
        dim = len(param_names)
        self._param_names = param_names[:]
        self._mu0 = mu0.copy()
        self._sigma0 = math.pow(np.linalg.det(cov0), 1.0 / 2.0 / dim)
        # Make the determinant of cov0 1 so that it agrees with the CMA-ES convention.
        self._cov0 = cov0.copy() / self._sigma0**2

    def _validate_user_prior(
        self, param_names: list[str], mu0: np.ndarray, cov0: np.ndarray
    ) -> None:
        dim = len(param_names)
        if dim != len(set(param_names)):
            raise ValueError(
                "Some elements in param_names are duplicated. Please make it a unique list."
            )
        if mu0.shape != (dim,) or cov0.shape != (dim, dim):
            raise ValueError(
                f"The shape of mu0 and cov0 must be (len(param_names)={dim}, ) and "
                f"(len(param_names)={dim}, len(param_names)={dim}), but got {mu0.shape} and "
                f"{cov0.shape}."
            )
        if not np.allclose(cov0, cov0.T):
            raise ValueError("cov0 must be a symmetric matrix.")
        if np.any(cov0 < 0.0):
            raise ValueError("All elements in cov0 must be non-negative.")
        if np.any(np.linalg.eigvals(cov0) < 0.0):
            raise ValueError("cov0 must be a semi-positive definite matrix.")

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if len(search_space) != 0 and set(search_space.keys()) != set(self._param_names):
            raise
        elif len(search_space) != 0:
            search_space = {
                param_name: search_space[param_name] for param_name in self._param_names
            }

        return super().sample_relative(study=study, trial=trial, search_space=search_space)

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
        direction: StudyDirection,
        population_size: int | None = None,
        randomize_start_point: bool = False,
    ) -> CmaClass:
        n_dimension = len(trans.bounds)
        mean = self._mu0.copy()
        cov = self._cov0.copy()

        # Avoid ZeroDivisionError in cmaes.
        sigma0 = max(self._sigma0, 1e-10)

        if self._with_margin:
            steps = np.empty(len(trans._search_space), dtype=float)
            for i, dist in enumerate(trans._search_space.values()):
                assert isinstance(dist, (IntDistribution, FloatDistribution))
                # Set step 0.0 for continuous search space.
                if dist.step is None or dist.log:
                    steps[i] = 0.0
                elif dist.low == dist.high:
                    steps[i] = 1.0
                else:
                    steps[i] = dist.step / (dist.high - dist.low)

            return cmaes.CMAwM(
                mean=mean,
                sigma=sigma0,
                bounds=trans.bounds,
                steps=steps,
                cov=cov,
                seed=self._cma_rng.rng.randint(1, 2**31 - 2),
                n_max_resampling=10 * n_dimension,
                population_size=population_size,
            )

        return cmaes.CMA(
            mean=mean,
            sigma=sigma0,
            cov=cov,
            bounds=trans.bounds,
            seed=self._cma_rng.rng.randint(1, 2**31 - 2),
            n_max_resampling=10 * n_dimension,
            population_size=population_size,
            lr_adapt=self._lr_adapt,
        )
