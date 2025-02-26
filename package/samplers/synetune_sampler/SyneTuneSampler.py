from __future__ import annotations

from collections.abc import Sequence
import datetime
from typing import Any
from typing import Dict

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub
from syne_tune.backend.trial_status import Status
from syne_tune.backend.trial_status import Trial
from syne_tune.backend.trial_status import TrialResult
from syne_tune.config_space import choice
from syne_tune.config_space import randint
from syne_tune.config_space import uniform
from syne_tune.optimizer.baselines import BORE
from syne_tune.optimizer.baselines import CQR
from syne_tune.optimizer.baselines import KDE
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune.optimizer.baselines import REA
from syne_tune.optimizer.scheduler import TrialScheduler


class AskTellScheduler:
    base_scheduler: TrialScheduler
    trial_counter: int
    completed_experiments: Dict[int, TrialResult]

    def __init__(self, base_scheduler: TrialScheduler):
        """
        Simple interface to use SyneTune schedulers in a custom loop, for example:

        .. code-block:: python

            scheduler = AskTellScheduler(base_scheduler=RandomSearch(config_space, metric=metric, mode=mode))
            for iter in range(max_iterations):
                trial_suggestion = scheduler.ask()
                test_result = target_function(**trial_suggestion.config)
                scheduler.tell(trial_suggestion, {metric: test_result})

        :param base_scheduler: Scheduler to be wrapped
        """
        self.base_scheduler = base_scheduler
        self.trial_counter = 0
        self.completed_experiments = {}

    def ask(self) -> Trial:
        """
        Ask the scheduler for new trial to run
        :return: Trial to run
        """
        trial_suggestion = self.base_scheduler.suggest(self.trial_counter)
        trial = Trial(
            trial_id=self.trial_counter,
            config=trial_suggestion.config,
            creation_time=datetime.datetime.now(),
        )
        self.trial_counter += 1
        return trial

    def tell(self, trial: Trial, experiment_result: Dict[str, float]) -> None:
        """
        Feed experiment results back to the Scheduler

        :param trial: Trial that was run
        :param experiment_result: {metric: value} dictionary with experiment results
        """
        trial_result = trial.add_results(
            metrics=experiment_result,
            status=Status.completed,
            training_end_time=datetime.datetime.now(),
        )
        self.base_scheduler.on_trial_complete(trial=trial, result=experiment_result)
        self.completed_experiments[trial_result.trial_id] = trial_result

    def best_trial(self, metric: str) -> TrialResult:
        """
        Return the best trial according to the provided metric.

        :param metric: Metric to use for comparison
        """
        if self.base_scheduler.mode == "max":
            sign = 1.0
        else:
            sign = -1.0

        return max(
            [value for key, value in self.completed_experiments.items()],
            key=lambda trial: sign * trial.metrics[metric],
        )


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
        # TODO pre-commit does not accept dict = None
        searcher_kwargs: dict | None = None,
    ) -> None:
        super().__init__(search_space)
        if searcher_kwargs is None:
            searcher_kwargs = {}
        self.metric = metric
        self.mode = mode
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
        # TODO Ask whether this conversion works as intended
        # TODO Optuna supports log and step arguments, these are not accountd for yet
        for name, dist in search_space.items():
            if isinstance(dist, FloatDistribution):
                syne_tune_space[name] = uniform(dist.low, dist.high)
            elif isinstance(dist, IntDistribution):
                syne_tune_space[name] = randint(dist.low, dist.high)
            elif isinstance(dist, CategoricalDistribution):
                syne_tune_space[name] = choice(dist.choices)
            else:
                raise ValueError(f"Unsupported distribution type: {type(dist)}")
        return syne_tune_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        trial_suggestion: Trial = self.scheduler.ask()
        return trial_suggestion.config

    def after_trial(
        self, study: Study, trial: FrozenTrial, state: TrialState, values: Sequence[float]
    ) -> None:
        # TODO Ask if there is a better way to get the trial object (this probably has a different creation time, right?)
        params = trial.params

        trial = Trial(
            trial_id=trial.number,
            config=params,
            creation_time=datetime.datetime.now(),
        )
        experiment_results = {self.metric: values[0]}

        self.scheduler.tell(trial=trial, experiment_result=experiment_results)
