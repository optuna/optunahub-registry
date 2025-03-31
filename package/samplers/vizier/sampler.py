from __future__ import annotations

from typing import Any
from typing import cast
from typing import Sequence

import optuna
from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub

from vizier.service import clients
from vizier.service import pyvizier as vz


class VizierSampler(optunahub.samplers.SimpleBaseSampler):
    problem: vz.ProblemStatement
    study_client: clients.Study | None = None
    suggestions: list[clients.Trial]

    def __init__(self, search_space: dict[str, BaseDistribution], seed: int | None = None):
        super().__init__(search_space, seed)
        self.problem = vz.ProblemStatement()
        for param_name, param_distribution in search_space.items():
            if isinstance(param_distribution, optuna.distributions.IntDistribution):
                self.problem.search_space.root.add_int_param(
                    param_name, param_distribution.low, param_distribution.high
                )
            elif isinstance(param_distribution, optuna.distributions.FloatDistribution):
                self.problem.search_space.root.add_float_param(
                    param_name, param_distribution.low, param_distribution.high
                )
            elif isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
                if not all(isinstance(choice, str) for choice in param_distribution.choices):
                    raise ValueError(
                        "Vizier only supports categorical distributions with string choices."
                    )
                self.problem.search_space.root.add_categorical_param(
                    param_name, cast(list[str], param_distribution.choices)
                )
            else:
                raise ValueError(f"Unsupported distribution: {param_distribution}")

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if self.study_client is None:
            self.problem.metric_information = list(
                vz.MetricInformation(
                    name=f"direction_{i}",
                    goal=vz.ObjectiveMetricGoal.MAXIMIZE
                    if direction == optuna.study.StudyDirection.MAXIMIZE
                    else vz.ObjectiveMetricGoal.MINIMIZE,
                )
                for i, direction in enumerate(study.directions)
            )

            study_config = vz.StudyConfig.from_problem(self.problem)
            study_config.algorithm = "DEFAULT"

            self.study_client = clients.Study.from_study_config(
                study_config, owner="owner", study_id=study.study_name
            )

        self.suggestions = self.study_client.suggest(count=1)

        ret = {}

        for key, value in self.suggestions[0].parameters.items():
            ret[key] = value
        return ret

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if values is not None:
            self.suggestions[0].complete(
                vz.Measurement(
                    {
                        f"direction_{i}": value
                        for i, value in enumerate(values)
                        if value is not None
                    }
                )
            )
