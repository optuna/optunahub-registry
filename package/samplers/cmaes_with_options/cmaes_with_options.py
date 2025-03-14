import cmaes
from optuna.samplers._cmaes import CmaEsSampler

from collections.abc import Callable
from collections.abc import Sequence
import copy
import math
import pickle
from typing import Any
from typing import cast
from typing import NamedTuple
from typing import TYPE_CHECKING
from typing import Union

import numpy as np

import optuna
from optuna import logging
from optuna._experimental import warn_experimental_argument
from optuna._imports import _LazyImport
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

_logger = logging.get_logger(__name__)

_EPS = 1e-10
# The value of system_attrs must be less than 2046 characters on RDBStorage.
_SYSTEM_ATTR_MAX_LENGTH = 2045

class CmaEsWithOptions(CmaEsSampler):
    def __init__(
        self,
        x0: dict[str, Any] | None = None,
        sigma0: float | None = None,
        n_startup_trials: int = 1,
        independent_sampler: BaseSampler | None = None,
        warn_independent_sampling: bool = True,
        seed: int | None = None,
        *,
        consider_pruned_trials: bool = False,
        restart_strategy: str | None = None,
        popsize: int | None = None,
        inc_popsize: int = 2,
        use_separable_cma: bool = False,
        with_margin: bool = False,
        lr_adapt: bool = False,
        source_trials: list[FrozenTrial] | None = None,
    ) -> None:
        # TODO(knshnb): Support sep-CMA-ES with margin.
        if use_separable_cma and with_margin:
            raise ValueError(
                "Currently, we do not support `use_separable_cma=True` and `with_margin=True`."
            )
        
        # TODO(c-bata): Support WS-sep-CMA-ES.
        if source_trials is not None and use_separable_cma:
            raise ValueError(
                "It is prohibited to pass `source_trials` argument when using separable CMA-ES."
            )

        if lr_adapt and (use_separable_cma or with_margin):
            raise ValueError(
                "It is prohibited to pass `use_separable_cma` or `with_margin` argument when "
                "using `lr_adapt`."
            )

        # TODO(knshnb): Support sep-CMA-ES with margin.
        if use_separable_cma and with_margin:
            raise ValueError(
                "Currently, we do not support `use_separable_cma=True` and `with_margin=True`."
            )

        
        self._use_separable_cma = use_separable_cma

        super().__init__(
            x0=x0,
            sigma0=sigma0,
            n_startup_trials=n_startup_trials,
            independent_sampler=independent_sampler,
            warn_independent_sampling=warn_independent_sampling,
            seed=seed,
            consider_pruned_trials=consider_pruned_trials,
            restart_strategy=restart_strategy,
            popsize=popsize,
            inc_popsize=inc_popsize,
            with_margin=with_margin,
            lr_adapt=lr_adapt,
            source_trials=source_trials,
        )

    def _attr_prefix(self) -> str:
        if self._use_separable_cma:
            return "sepcma:"
        elif self._with_margin:
            return "cmawm:"
        else:
            return "cma:"

    def _init_optimizer(
        self,
        trans: _SearchSpaceTransform,
        direction: StudyDirection,
        population_size: int | None = None,
        randomize_start_point: bool = False,
    ) -> "CmaClass":
        mean, sigma0, cov, bounds, n_dimension, population_size = self._get_cmaes_params(
            trans, direction, population_size, randomize_start_point
        )

        if self._use_separable_cma:
            return cmaes.SepCMA(
                mean=mean,
                sigma=sigma0,
                bounds=trans.bounds,
                seed=self._cma_rng.rng.randint(1, 2**31 - 2),
                n_max_resampling=10 * n_dimension,
                population_size=population_size,
            )

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