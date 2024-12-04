from __future__ import annotations

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import pandas as pd


class HEBOSampler(BaseSampler):  # type: ignore
    def __init__(
        self,
        seed: int | None = None,
        constant_liar: bool = False,
        independent_sampler: BaseSampler | None = None,
    ) -> None:
        self._seed = seed
        self._intersection_search_space = IntersectionSearchSpace()
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._constant_liar = constant_liar

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, float]:
        if self._constant_liar:
            target_states = [TrialState.COMPLETE, TrialState.RUNNING]
        else:
            target_states = [TrialState.COMPLETE]
        trials = study.get_trials(deepcopy=False, states=target_states)
        if len([t for t in trials if t.state == TrialState.COMPLETE]) < 1:
            return {}

        # Assume that the back-end HEBO implementation aims to minimize.
        if study.direction == StudyDirection.MINIMIZE:
            worst_values = max(t.values for t in trials if t.state == TrialState.COMPLETE)
        else:
            worst_values = min(t.values for t in trials if t.state == TrialState.COMPLETE)
        sign = 1.0 if study.direction == StudyDirection.MINIMIZE else -1.0

        hebo = HEBO(self._convert_to_hebo_design_space(search_space))
        for t in trials:
            hebo_params = {name: t.params[name] for name in search_space.keys()}
            if t.state == TrialState.COMPLETE:
                hebo.observe(pd.DataFrame([hebo_params]), np.asarray([t.values * sign]))
            elif t.state == TrialState.RUNNING:
                # If `constant_liar == True`, assume that the RUNNING params result in bad values,
                # thus preventing the simultaneous suggestion of (almost) the same params
                # during parallel execution.
                hebo.observe(pd.DataFrame([hebo_params]), np.asarray([worst_values]))
            else:
                assert False
        params_pd = hebo.suggest()
        params = {}
        for name in search_space.keys():
            params[name] = params_pd[name].to_numpy()[0]
        return params

    def _convert_to_hebo_design_space(
        self, search_space: dict[str, BaseDistribution]
    ) -> DesignSpace:
        design_space = []
        for name, distribution in search_space.items():
            if isinstance(distribution, FloatDistribution) and not distribution.log:
                design_space.append(
                    {
                        "name": name,
                        "type": "num",
                        "lb": distribution.low,
                        "ub": distribution.high,
                    }
                )
            elif isinstance(distribution, FloatDistribution) and distribution.log:
                design_space.append(
                    {
                        "name": name,
                        "type": "pow",
                        "lb": distribution.low,
                        "ub": distribution.high,
                    }
                )
            elif isinstance(distribution, IntDistribution) and distribution.log:
                design_space.append(
                    {
                        "name": name,
                        "type": "pow_int",
                        "lb": distribution.low,
                        "ub": distribution.high,
                    }
                )
            elif isinstance(distribution, IntDistribution) and distribution.step:
                design_space.append(
                    {
                        "name": name,
                        "type": "step_int",
                        "lb": distribution.low,
                        "ub": distribution.high,
                        "step": distribution.step,
                    }
                )
            elif isinstance(distribution, IntDistribution):
                design_space.append(
                    {
                        "name": name,
                        "type": "int",
                        "lb": distribution.low,
                        "ub": distribution.high,
                    }
                )
            elif isinstance(distribution, CategoricalDistribution):
                design_space.append(
                    {
                        "name": name,
                        "type": "cat",
                        "categories": distribution.choices,
                    }
                )
            else:
                raise NotImplementedError(f"Unsupported distribution: {distribution}")
        return DesignSpace().parse(design_space)

    def infer_relative_search_space(self, study, trial):  # type: ignore
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

    def sample_independent(self, study, trial, param_name, param_distribution):  # type: ignore
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
