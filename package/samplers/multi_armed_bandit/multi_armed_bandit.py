from collections import defaultdict
from typing import Any
from typing import Optional

from optuna.distributions import BaseDistribution
from optuna.samplers import RandomSampler
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class MultiArmedBanditSampler(RandomSampler):
    """Sampler based on Multi-armed Bandit Algorithm.

    Args:
        epsilon (float):
            Params for epsolon-greedy algorithm.
            epsilon is probability of selecting arm randomly.
        seed (int):
            Seed for random number generator and arm selection.

    """

    def __init__(
        self,
        epsilon: float = 0.7,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed)
        self._epsilon = epsilon

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if self._rng.rng.rand() < self._epsilon:
            return self._rng.rng.choice(param_distribution.choices)
        else:
            states = (TrialState.COMPLETE, TrialState.PRUNED)
            trials = study._get_trials(deepcopy=False, states=states, use_cache=True)

            rewards_by_choice: defaultdict = defaultdict(float)
            cnt_by_choice: defaultdict = defaultdict(int)
            for t in trials:
                rewards_by_choice[t.params[param_name]] += t.value
                cnt_by_choice[t.params[param_name]] += 1

            if study.direction == StudyDirection.MINIMIZE:
                return min(
                    param_distribution.choices,
                    key=lambda x: rewards_by_choice[x] / max(cnt_by_choice[x], 1),
                )
            else:
                return max(
                    param_distribution.choices,
                    key=lambda x: rewards_by_choice[x] / max(cnt_by_choice[x], 1),
                )
