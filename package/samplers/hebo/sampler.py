from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub
import pandas as pd

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO


class HEBOSampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(self, search_space: dict[str, BaseDistribution]) -> None:
        super().__init__(search_space)
        self._hebo = HEBO(self._convert_to_hebo_design_space(search_space))

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, float]:
        params_pd = self._hebo.suggest()

        params = {}
        for name in search_space.keys():
            params[name] = params_pd[name].to_numpy()[0]
        return params

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._hebo.observe(pd.DataFrame([trial.params]), np.asarray([values]))

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
