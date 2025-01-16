from __future__ import annotations

import math
from typing import Any
from typing import Union

import cmaes
import numpy as np
from optuna import Study
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


CmaClass = Union[cmaes.CMA, cmaes.SepCMA, cmaes.CMAwM]


class UserPriorCmaEsSampler(CmaEsSampler):
    """A sampler using `cmaes <https://github.com/CyberAgentAILab/cmaes>`__ as the backend with user prior.

    Please check ``CmaEsSampler`` in Optuna for more details of each argument.
    This class modified the arguments ``x0`` and ``sigma0`` in ``CmaEsSampler`` of Optuna.
    Furthermore, due to the incompatibility,
    This class does not support ``source_trials`` and ``use_separable_cma``.

    Args:
        param_names:
            The list of the parameter names to be tuned. This list must be a unique list.
        mu0:
            The mean vector used for the initialization of CMA-ES.
        cov0:
            The covariance matrix used for the initialization of CMA-ES.
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
        self._param_names = param_names[:]
        self._mu0 = mu0.astype(float)
        self._cov0 = cov0.astype(float)

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
        if np.any(np.linalg.eigvals(cov0) < 0.0):
            raise ValueError("cov0 must be a semi-positive definite matrix.")

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if len(search_space) != 0 and set(search_space.keys()) != set(self._param_names):
            raise ValueError(
                "The keys in search_space and param_names did not match. "
                "The most probable reason is duplicated names in param_names."
            )
        elif len(search_space) != 0:
            # Ensure the parameter order is identical to that in param_names.
            search_space = {
                param_name: search_space[param_name] for param_name in self._param_names
            }

        return super().sample_relative(study=study, trial=trial, search_space=search_space)

    def _calculate_initial_params(
        self, trans: _SearchSpaceTransform
    ) -> tuple[np.ndarray, float, np.ndarray]:
        # NOTE(nabenabe): Except this method, everything is basically based on Optuna v4.0.0.
        # As this class does not support some cases supported by Optuna, I simply added validation
        # to each method, but otherwise, nothing changed. In principle, if users find a bug, it is
        # likely that the bug exists in this method.
        search_space = trans._search_space.copy()
        if any(
            not isinstance(d, (IntDistribution, FloatDistribution)) for d in search_space.values()
        ):
            raise ValueError("search_space cannot include categorical parameters.")
        if any(
            d.log
            for d in search_space.values()
            if isinstance(d, (FloatDistribution, IntDistribution))
        ):
            src_url = "https://hub.optuna.org/samplers/user_prior_cmaes/"
            raise ValueError(
                "search_space for user_prior cannot include log scale. "
                f"Please use the workaround described in {src_url}."
            )

        dim = len(self._param_names)
        raw_bounds = trans._raw_bounds
        domain_sizes = raw_bounds[:, 1] - raw_bounds[:, 0]
        is_single = domain_sizes == 0.0

        mu0 = self._mu0.copy()
        mu0[is_single] = 0.5
        # Clip into [0, 1].
        mu0[~is_single] = (mu0[~is_single] - raw_bounds[~is_single, 0]) / domain_sizes[~is_single]

        # We also need to transform the covariance matrix accordingly to adapt to the [0, 1] scale.
        cov0 = self._cov0 / (domain_sizes * domain_sizes[:, np.newaxis])

        # Make the determinant of cov0 1 so that it agrees with the CMA-ES convention.
        sigma0 = math.pow(np.linalg.det(cov0), 1.0 / 2.0 / dim)
        # Avoid ZeroDivisionError in cmaes.
        sigma0 = max(sigma0, 1e-10)
        cov0 /= sigma0**2

        return mu0, sigma0, cov0

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
        direction: StudyDirection,
        population_size: int | None = None,
        randomize_start_point: bool = False,
    ) -> CmaClass:
        n_dimension = len(trans.bounds)
        mu0, sigma0, cov0 = self._calculate_initial_params(trans)

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
                mean=mu0,
                sigma=sigma0,
                bounds=trans.bounds,
                steps=steps,
                cov=cov0,
                seed=self._cma_rng.rng.randint(1, 2**31 - 2),
                n_max_resampling=10 * n_dimension,
                population_size=population_size,
            )

        return cmaes.CMA(
            mean=mu0,
            sigma=sigma0,
            cov=cov0,
            bounds=trans.bounds,
            seed=self._cma_rng.rng.randint(1, 2**31 - 2),
            n_max_resampling=10 * n_dimension,
            population_size=population_size,
            lr_adapt=self._lr_adapt,
        )
