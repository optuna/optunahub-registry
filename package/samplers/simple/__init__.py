import abc
from typing import Any

from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial


class SimpleSampler(BaseSampler, abc.ABC):
    def __init__(self, search_space: dict[str, BaseDistribution]):
        self.search_space = search_space

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        # This method is optional.
        # If you want to optimize the function with the eager search space,
        # please implement this method.
        return self.search_space

    @abc.abstractmethod
    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        # This method is required.
        # This method is called at the beginning of each trial in Optuna to sample parameters.
        raise NotImplementedError

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # This method is optional.
        # If you want to treat the parameters which are not include in the relative search space,
        # please implement this method.
        raise NotImplementedError
