from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
import optunahub
from syne_tune.backend.trial_status import Trial as SyneTuneTrial
from syne_tune.config_space import choice
from syne_tune.config_space import finrange
from syne_tune.config_space import lograndint
from syne_tune.config_space import loguniform
from syne_tune.config_space import randint
from syne_tune.config_space import uniform
from syne_tune.optimizer.baselines import BORE
from syne_tune.optimizer.baselines import CQR
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.optimizer.baselines import REA


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial
    from optuna.trial import TrialState


# Currently supported methods
scheduler_cls_dict = {
    "RandomSearch": RandomSearch,
    "BORE": BORE,
    "REA": REA,
    "CQR": CQR,
}


class SyneTuneSampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(
        self,
        metric: str,
        search_space: dict[str, BaseDistribution],
        direction: str = "minimize",
        searcher_method: str = "CQR",
        searcher_kwargs: dict | None = None,
    ) -> None:
        """
        A sampler that uses SyneTune v0.14.2 or greater.

        Please check the API reference for more details:
            https://syne-tune.readthedocs.io/en/latest/_apidoc/modules.html

        Args:
            search_space:
                A dictionary of Optuna distributions.
            metric:
                The metric to optimize.
            searcher_method:
                The search method to be run. Currently supported searcher methods are: BORE, CQR,
                REA, RandomSearch.
            direction:
                Direction of optimization, either "minimize" or "maximize". Defaults to "minimize".
            searcher_kwargs: Additional keyword arguments for the searcher. Defaults to None.
        """
        super().__init__(search_space)

        if direction not in ["minimize", "maximize"]:
            raise ValueError(
                f"direction must be `minimize` or `maximize`, but got direction={direction}."
            )
        searcher_cls = scheduler_cls_dict.get(searcher_method)
        if searcher_cls is None:
            avail_searcher_method = list(scheduler_cls_dict.keys())
            raise ValueError(
                f"searcher_method must be in {avail_searcher_method}, but got {searcher_method}."
            )

        self._metric = metric
        self._trial_mapping: dict[int, Any] = {}
        self._syne_tune_space = self._convert_optuna_to_syne_tune(search_space)
        searcher_kwargs = searcher_kwargs or {}
        self._do_minimize = direction == "minimize"
        if searcher_method == "RandomSearch":
            scheduler = RandomSearch(
                config_space=self._syne_tune_space,
                metrics=[self._metric],
                do_minimize=self._do_minimize,
                **searcher_kwargs,
            )
        else:
            scheduler = searcher_cls(
                config_space=self._syne_tune_space,
                metric=self._metric,
                do_minimize=self._do_minimize,
                **searcher_kwargs,
            )
        self._scheduler = scheduler

    def _convert_optuna_to_syne_tune(
        self, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        syne_tune_space = {}
        for name, distribution in search_space.items():
            if isinstance(distribution, CategoricalDistribution):
                syne_tune_space[name] = choice(distribution.choices)
            elif isinstance(distribution, IntDistribution):
                low, high = distribution.low, distribution.high
                if distribution.log:
                    syne_tune_space[name] = lograndint(low, high)
                elif distribution.step == 1:
                    syne_tune_space[name] = randint(low, high)
                else:
                    size = (high - low) // distribution.step + 1
                    syne_tune_space[name] = finrange(low, high, size=size, cast_int=True)
            elif isinstance(distribution, FloatDistribution):
                low, high = distribution.low, distribution.high
                if not distribution.log:
                    if distribution.step is not None:
                        size = int(np.round((high - low) / distribution.step)) + 1
                        syne_tune_space[name] = finrange(low, high, size=size, cast_int=False)
                    else:
                        syne_tune_space[name] = uniform(low, high)
                elif distribution.step is None:
                    syne_tune_space[name] = loguniform(low, high)
                else:
                    raise ValueError("`step` cannot be specified if `log=True`.")
            else:
                raise NotImplementedError(f"Unknown Hyperparameter Type: {type(distribution)}")
        return syne_tune_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        do_minimize = study.direction.name == "MINIMIZE"
        if self._do_minimize != do_minimize:
            raise ValueError("The direction mismatch in `SyneTuneSampler` and `create_study`.")

        trial_suggestion = self._scheduler.suggest()
        trial_suggestion = SyneTuneTrial(
            trial_id=trial.number,
            config=trial_suggestion.config,
            creation_time=trial.datetime_start,
        )
        self._trial_mapping[trial.number] = trial_suggestion
        return trial_suggestion.config

    def after_trial(
        self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float]
    ) -> None:
        self._scheduler.on_trial_complete(
            self._trial_mapping[trial.number], result={self._metric: values[0]}
        )
