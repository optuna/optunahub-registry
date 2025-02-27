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
    It defaults to the base sampler for other variables.
    There is one additional parameter: `burn_in`, which allows to establish the values of the categorical
    choices during a burn-in phase in which the sampler runs through each choice `burn_in` number of times.
    Thereafter, it draws randomly from the categories and chooses a winner based on these random draws.
    This works only for a single categorical variable but could be extended straightforwardly if desired.
    """

    def __init__(
        self,
        base_sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(),
        burn_in: int = 10,
        cat_var_name: Optional[str] = None,
    ) -> None:
        """

        :param base_sampler: a valid optuna sampler
        :param burn_in: an integer setting the duration of the burn in phase
        :param cat_var_name: you can name the categorical variable here if you wish (if not, it will be discovered
        during the sampling phase)
        """
        self.base_sampler = base_sampler
        self.burn_in, self.burning_in = burn_in, True

    def infer_relative_search_space(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> Dict[str, optuna.distributions.BaseDistribution]:
        return self.base_sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: Dict[str, optuna.distributions.BaseDistribution],
    ) -> Dict[str, Any]:
        return self.base_sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions,
    ) -> Dict[str, Any]:
        if isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
            return self.sample_categorical(study, param_distribution)
        return self.base_sampler.sample_independent(study, trial, param_name, param_distribution)

    def sample_categorical(
        self, study: optuna.study.Study, param_distribution: optuna.distributions
    ) -> Dict[str, Any]:
        """
        This performs `burn_in` number of trials for _each categorical choice_ sequentially before moving to Thompson
        sampling. This transition is logged.

        For each sample in the Thompson sampling phase, a random sample is drawn (with replacement) from each category
        The category with the best random draw is selected as the choice for this trial
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
                study.cat_dict, key=lambda x: self.base_sampler._rng.rng.choice(study.cat_dict[x])
            )
        return cat_choice

    def after_trial(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Sequence[float],
    ) -> None:
        if not hasattr(study, "cat_var_name"):
            # determine the categorical parameter name if it hasn't been discovered yet
            for param in trial.distributions:
                if isinstance(
                    trial.distributions[param], optuna.distributions.CategoricalDistribution
                ):
                    study.cat_var_name = param
        if not hasattr(study, "cat_dict"):
            study.cat_dict = defaultdict(list)
        cur_cat = trial.params[study.cat_var_name]
        cur_val = values[0]
        study.cat_dict[cur_cat].append(cur_val)
