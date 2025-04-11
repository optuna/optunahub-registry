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


def _add_search_space_to_problem(
    problem: vz.ProblemStatement, search_space: dict[str, BaseDistribution]
) -> None:
    for param_name, param_distribution in search_space.items():
        if isinstance(param_distribution, optuna.distributions.IntDistribution):
            problem.search_space.root.add_int_param(
                param_name, param_distribution.low, param_distribution.high
            )
        elif isinstance(param_distribution, optuna.distributions.FloatDistribution):
            problem.search_space.root.add_float_param(
                param_name, param_distribution.low, param_distribution.high
            )
        elif isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
            if all(isinstance(choice, (int, float)) for choice in param_distribution.choices):
                problem.search_space.root.add_discrete_param(
                    param_name, cast(list[int | float], param_distribution.choices)
                )
                return
            if all(isinstance(choice, str) for choice in param_distribution.choices):
                problem.search_space.root.add_categorical_param(
                    param_name, cast(list[str], param_distribution.choices)
                )
                return
            raise ValueError(
                f"Vizier only supports int, float, or str choices in CategoricalDistribution. "
                f"Got {param_distribution.choices}."
            )
        else:
            raise ValueError(f"Vizier sampler does not support {param_distribution}.")


class VizierSampler(optunahub.samplers.SimpleBaseSampler):
    search_space: dict[str, BaseDistribution] | None = None
    suggestions: list[clients.Trial] | None = None

    def __init__(
        self, algorithm: str = "DEFAULT", search_space: dict[str, BaseDistribution] | None = None
    ):
        super().__init__(search_space)
        self.algorithm: str = algorithm
        self.study_client: clients.Study | None = None

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        # The first trial is skipped when search_space is empty.
        if len(search_space) == 0:
            return {}

        if self.search_space is None:
            self.search_space = search_space

        if self.study_client is None:
            problem = vz.ProblemStatement()
            _add_search_space_to_problem(problem, search_space)
            problem.metric_information = list(
                vz.MetricInformation(
                    name=f"direction_{i}",
                    goal=vz.ObjectiveMetricGoal.MAXIMIZE
                    if direction == optuna.study.StudyDirection.MAXIMIZE
                    else vz.ObjectiveMetricGoal.MINIMIZE,
                )
                for i, direction in enumerate(study.directions)
            )

            study_config = vz.StudyConfig.from_problem(problem)
            study_config.algorithm = self.algorithm

            self.study_client = clients.Study.from_study_config(
                study_config, owner="owner", study_id=study.study_name
            )

        self.suggestions = self.study_client.suggest(count=1)

        ret: dict[str, Any] = {}
        if self.suggestions is None:
            assert False, "unreachable"
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
        if values is not None and self.suggestions is not None:
            self.suggestions[0].complete(
                vz.Measurement(
                    {
                        f"direction_{i}": value
                        for i, value in enumerate(values)
                        if value is not None
                    }
                )
            )
