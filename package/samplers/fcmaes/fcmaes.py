from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from uuid import uuid4

import numpy as np
import optuna
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import pandas as pd
from scipy.optimize import Bounds

from fcmaes.cmaescpp import ACMA_C


def estimate_non_zero_parameters(trials: List[FrozenTrial]) -> Optional[np.ndarray]:
    threshold = 0.0

    values = [
        trial.value
        for trial in trials
        if trial.state == TrialState.COMPLETE
        and isinstance(trial.value, float)
        and trial.value > 0.0
    ]

    if values:
        threshold = np.percentile(values, 75)

        params: List[Dict] = [
            trial.params
            for trial in trials
            if isinstance(trial.value, float) and trial.value > threshold
        ]

        df = pd.DataFrame.from_dict(params)  # type: ignore

        mean = df.mean()

        return mean.values  # type:ignore

    return None


class _AttrKeys(NamedTuple):
    optimizer: Callable[[], str]
    generation: Callable[[], str]


class FastCmaesSampler(BaseSampler):
    def __init__(
        self,
        popsize: int,
        search_space: dict[str, BaseDistribution] | None = None,
        seed: int = 42,
    ):
        self.signature = str(uuid4())
        self.popsize = popsize
        self.search_space = search_space
        self.seed = seed
        self.optimizer: Optional[ACMA_C] = None

        self._intersection_search_space = IntersectionSearchSpace()

        self.iterations = 0

        self.ask_queue: List[Dict] = []

    def _init_optimizer(self, x0: np.ndarray, bounds: Bounds, popsize: int) -> ACMA_C:
        return ACMA_C(len(bounds.lb), bounds, x0=x0, popsize=popsize, input_sigma=0.5)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        if self.search_space is not None:
            return self.search_space

        search_space: Dict[str, BaseDistribution] = {}
        for name, distribution in self._intersection_search_space.calculate(study).items():
            if distribution.single():
                continue
            search_space[name] = distribution

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        if len(self.ask_queue) > 0:
            return self.ask_queue.pop()

        trans = _SearchSpaceTransform(search_space, transform_0_1=True)

        bounds = Bounds(trans.bounds[:, 0].flatten(), trans.bounds[:, 1].flatten())  # type: ignore

        completed_trials = self._get_trials(study)

        if self.optimizer is None:
            self.optimizer = self._init_optimizer(
                estimate_non_zero_parameters(study.trials),  # type: ignore
                bounds,
                popsize=self.popsize,
            )

        solution_trials = self._get_solution_trials(completed_trials, self.iterations)

        if len(solution_trials) >= self.popsize:
            # Prepare solutions list
            solutions: List = []
            values: List = []

            for t in solution_trials[: self.popsize]:
                assert t.value is not None, "completed trials must have a value"
                # Convert Optuna's representation to cmaes.CatCma's internal representation.

                value = t.value if study.direction == StudyDirection.MINIMIZE else -t.value

                solution = trans.transform(t.params)

                solutions.append(solution)
                values.append(value)

            self.optimizer.tell(np.array(values), np.array(solutions))

            self.iterations += 1

        solution = self.optimizer.ask()

        generation_attr_key = self._attr_keys.generation()

        study._storage.set_trial_system_attr(trial._trial_id, generation_attr_key, self.iterations)

        for row in solution:
            self.ask_queue.append(trans.untransform(row))

        return self.ask_queue.pop()

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        independent_sampler = optuna.samplers.RandomSampler()
        return independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def _get_solution_trials(
        self, trials: List[FrozenTrial], generation: int
    ) -> List[FrozenTrial]:
        generation_attr_key = self._attr_keys.generation()
        return [t for t in trials if generation == t.system_attrs.get(generation_attr_key, -1)]

    @property
    def _attr_keys(self) -> _AttrKeys:
        def optimizer_key_template() -> str:
            return self.signature + "optimizer"

        def generation_attr_key_template() -> str:
            return self.signature + "generation"

        return _AttrKeys(
            optimizer_key_template,
            generation_attr_key_template,
        )

    def _get_trials(self, study: optuna.Study) -> List[FrozenTrial]:
        complete_trials = []
        for t in study._get_trials(deepcopy=False, use_cache=True):
            if t.state == TrialState.COMPLETE:
                complete_trials.append(t)
        return complete_trials
