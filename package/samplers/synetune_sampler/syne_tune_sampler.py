from __future__ import annotations

from collections.abc import Sequence
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
from syne_tune.optimizer.baselines import KDE
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.optimizer.baselines import REA
from syne_tune.optimizer.scheduler import AskTellScheduler


# Currently supported methods
searcher_cls_dict = {
    "random_search": RandomSearch,
    "bore": BORE,
    "kde": KDE,
    "regularized_evolution": REA,
    "cqr": CQR,
}


class SyneTuneSampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(
        self,
        search_space: dict[str, BaseDistribution],
        mode: str = "min",
        metric: str = "mean_loss",
        searcher_method: str = "random_search",
        trial_mapping: Optional[dict] = None,
        searcher_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__(search_space)
        if searcher_kwargs is None:
            searcher_kwargs = {}
        if trial_mapping is None:
            trial_mapping = {}
        self.metric = metric
        self.mode = mode
        self.trial_mapping = trial_mapping
        self._syne_tune_space = self._convert_optuna_to_syne_tune(search_space)
        self.scheduler = AskTellScheduler(
            base_scheduler=searcher_cls_dict[searcher_method](
                self._syne_tune_space, metric=self.metric, mode=self.mode, **searcher_kwargs
            )
        )

    def _convert_optuna_to_syne_tune(
        self, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        syne_tune_space = {}
        for name, distribution in search_space.items():
            if isinstance(distribution, CategoricalDistribution):
                syne_tune_space[name] = choice(distribution.choices)
            elif distribution.step is not None:
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
        trial_suggestion: Trial = self.scheduler.ask()
        self.trial_mapping[trial_suggestion.trial_id] = trial_suggestion
        return trial_suggestion.config

    def after_trial(
        self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float]
    ) -> None:
        params = trial.params

        trial = Trial(
            trial_id=trial.number,
            config=params,
            creation_time=self.trial_mapping[trial.number].creation_time,
        )
        experiment_results = {self.metric: values[0]}

        self.scheduler.tell(trial=trial, experiment_result=experiment_results)
