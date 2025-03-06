from __future__ import annotations

from collections import defaultdict
import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import optuna


logger = logging.getLogger(__name__)


class ThompsonSampler(optuna.samplers.BaseSampler):
    """
    This subclasses optuna.sampler.BaseSampler to add Thompson sampling for categorical variables.
    It defaults to a specified `base_sampler` for other variables.

    There is one additional keyword argument: `burn_in`, which establishes the values of the categorical
    choices during a burn-in phase in which the sampler runs through each choice `burn_in` number of times.
    Thereafter, it draws randomly from the categories and chooses a winner based on these random draws.
    This version works only for a single categorical variable but could be extended if desired.
    """

    def __init__(
        self,
        burn_in: int = 10,
        base_sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(),
        categorical_variable_name: Optional[str] = None,
    ) -> None:
        """

        :param base_sampler: a valid optuna sampler
        :param burn_in: an integer setting the duration of the burn in phase
        :param categorical_variable_name: you can name the categorical variable here if you wish (if not, it will be discovered
        during the sampling phase)
        """
        self.base_sampler = base_sampler
        self.burn_in, self.burning_in = burn_in, True
        self.categorical_variable_name = categorical_variable_name

    def infer_relative_search_space(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> Dict[str, optuna.distributions.BaseDistribution]:
        """

        Args:
            study: optuna.study.Study instance
            trial: optuna.trial.FrozenTrial instance

        Returns:
            falls back to the behavior of `self.base_sampler.infer_relative_search_space`
        """
        return self.base_sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: Dict[str, optuna.distributions.BaseDistribution],
    ) -> Dict[str, Any]:
        """

        Args:
            study: optuna.study.Study instance
            trial: optuna.trial.FrozenTrial instance
            search_space: dictionary of variable names and optuna.distributions.BaseDistribution

        Returns:
            falls back to the behavior of `self.base_sampler.infer_relative_search_space`
        """
        return self.base_sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions,
    ) -> Dict[str, Any]:
        """

        Args:
            study: optuna.study.Study instance
            trial: optuna.trial.FrozenTrial instance
            param_name: name of parameter being sampled
            param_distribution: distribution of the parameter being sampled

        Returns:
            If `param_distribution` is an instance of an optuna.distributions.CategoricalDistribution
            then this performs the Thompson sampling algorithm impleneted in `sample_categorical`.
            Otherwise, this falls back to the behavior of `base_sampler.sample_independent`.
        """
        if isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
            return self.sample_categorical(study, param_distribution)
        return self.base_sampler.sample_independent(study, trial, param_name, param_distribution)

    def sample_categorical(
        self, study: optuna.study.Study, param_distribution: optuna.distributions
    ) -> Dict[str, Any]:
        """
        This performs `self.burn_in` number of trials for _each categorical choice_ sequentially before moving to
        Thompson sampling. This transition is logged.

        For each sample in the Thompson sampling phase, a random sample is drawn (with replacement) from each category
        The category with the best random draw is selected as the choice for this trial

        Args:
            study: optuna.study.Study instance
            param_distribution: distribution of the parameter being sampled

        Returns:
            for the Nth trial before the burn-in phase is complete, after the ((self.burn_in) * len(categories))th
            trial, this returns the categorical choice whose value is N % len(categories). After the burn-in phase is
            complete, this draws a random sample from each of the categories and returns the category that
            provided the maximum sample. This follows the algorithm in arxiv:1707.02038
        """
        categories = param_distribution.choices
        if len(study.trials) <= (self.burn_in) * len(categories):
            cat_choice = categories[int(len(study.trials) % len(categories))]
        else:
            if self.burning_in:
                logger.info(
                    f"{self.burn_in} burns for each of the {len(categories)} categories have been completed. Moving to Thompson sampling now."
                )
                self.burning_in = False
            cat_choice = max(
                study.categorical_variable_samples,
                key=lambda x: self.base_sampler._rng.rng.choice(
                    study.categorical_variable_samples[x]
                ),
            )
        return cat_choice

    def after_trial(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Sequence[float],
    ) -> None:
        """
        adds trials to the `categorical_variable_samples` dictionary of `study`. Safely creates this dictionary if
        it does not already exist.

        Args:
            study: optuna.study.Study instance
            trial: optuna.trial.FrozenTrial instance
            state: optuna.trial.TrialState instance
            values: the list of values sampled
        """
        for param in trial.distributions:
            if isinstance(
                trial.distributions[param], optuna.distributions.CategoricalDistribution
            ):
                if not hasattr(study, "categorical_variable_name"):
                    # determine the categorical parameter name if it hasn't been discovered yet
                    study.categorical_variable_name = param
                if not hasattr(study, "categorical_variable_samples"):
                    study.categorical_variable_samples = defaultdict(list)
                cur_category = trial.params[study.categorical_variable_name]
                cur_val = values[0]
                study.categorical_variable_samples[cur_category].append(cur_val)
