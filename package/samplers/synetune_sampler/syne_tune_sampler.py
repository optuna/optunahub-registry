from __future__ import annotations

from collections.abc import Sequence
import datetime
from typing import Any
from typing import Optional

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub
from syne_tune.backend.trial_status import Trial
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
        searcher_kwargs: Optional[dict] = None,
    ) -> None:
        """
         A sampler that uses SyneTune v0.13.0 or greater.

        Please check the API reference for more details:
            https://syne-tune.readthedocs.io/en/latest/_apidoc/modules.html

        Args:
            search_space:
                A dictionary of Optuna distributions.
            metric:
                The metric to optimize.
            searcher_method:
                The search method to be run. Currently supported searcher methods are: BORE, CQR, KDE, REA, RandomSearch.
            direction:
                Direction of optimization, either "minimize" or "maximize". Defaults to "minimize".
            searcher_kwargs: Additional keyword arguments for the searcher. Defaults to None.
        """
        super().__init__(search_space)
        assert direction in ["minimize", "maximize"]
        self.direction = direction
        self.metric = metric
        self.trial_mapping: dict[int, Any] = {}
        self._syne_tune_space = self._convert_optuna_to_syne_tune(search_space)

        searcher_kwargs = searcher_kwargs or {}

        if searcher_method == "RandomSearch":
            scheduler = scheduler_cls_dict[searcher_method](
                config_space=self._syne_tune_space,
                metrics=[self.metric],
                do_minimize=(self.direction == "minimize"),
                **searcher_kwargs,
            )
        else:
            scheduler = scheduler_cls_dict[searcher_method](
                config_space=self._syne_tune_space,
                metric=self.metric,
                do_minimize=(self.direction == "minimize"),
                **searcher_kwargs,
            )
        self.scheduler = scheduler

    def _convert_optuna_to_syne_tune(
        self, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        syne_tune_space = {}
        for name, distribution in search_space.items():
            if isinstance(distribution, CategoricalDistribution):
                syne_tune_space[name] = choice(distribution.choices)
            elif distribution.step is not None:
                if distribution.log:
                    raise ValueError(
                        f"{self.__class__.__name__} cannot specify not None `step` and `log=True` simultaneously."
                    )
                if isinstance(distribution, IntDistribution):
                    syne_tune_space[name] = finrange(
                        lower=distribution.low,
                        upper=distribution.high,
                        size=distribution.step,
                        cast_int=True,
                    )
                elif isinstance(distribution, FloatDistribution):
                    syne_tune_space[name] = finrange(
                        lower=distribution.low,
                        upper=distribution.high,
                        size=distribution.step,
                        cast_int=False,
                    )
            elif distribution.step is None and not distribution.log:
                if isinstance(distribution, IntDistribution):
                    syne_tune_space[name] = randint(
                        lower=distribution.low, upper=distribution.high
                    )
                elif isinstance(distribution, FloatDistribution):
                    syne_tune_space[name] = uniform(
                        lower=distribution.low, upper=distribution.high
                    )
            elif distribution.log:
                if isinstance(distribution, FloatDistribution):
                    syne_tune_space[name] = loguniform(distribution.low, distribution.high)
                else:
                    syne_tune_space[name] = lograndint(distribution.low, distribution.high)
            else:
                raise NotImplementedError(f"Unknown Hyperparameter Type: {type(distribution)}")
        return syne_tune_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if (self.direction == "minimize" and study.direction.name != "MINIMIZE") or (
            self.direction == "maximize" and study.direction.name != "MAXIMIZE"
        ):
            raise ValueError(
                f"The direction mismatch in {self.__class__.__name__} and `create_study`."
            )

        trial_suggestion = self.scheduler.suggest()
        trial_suggestion = Trial(
            trial_id=trial.number,
            config=trial_suggestion.config,
            creation_time=datetime.datetime.now(),
        )
        self.trial_mapping[trial.number] = trial_suggestion
        return trial_suggestion.config

    def after_trial(
        self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float]
    ) -> None:
        self.scheduler.on_trial_complete(
            self.trial_mapping[trial.number], result={self.metric: values[0]}
        )
