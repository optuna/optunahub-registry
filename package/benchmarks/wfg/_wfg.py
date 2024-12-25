from __future__ import annotations

from typing import Any

import optproblems.wfg
import optuna
import optunahub


class Problem(optunahub.benchmarks.BaseProblem):
    """Wrapper class for the WFG test suite of optproblems."""

    def __init__(
        self, function_id: int, num_objectives: int, num_variables: int, k: int, **kwargs: Any
    ) -> None:
        """Initialize the problem.
        Args:
            function_id: Function ID of the WFG problem in [1, 9].
            num_objectives: Number of objectives.
            num_variables: Number of variables.
            k: Number of position parameters. It must hold k < num_variables and k must be a multiple of num_objectives - 1. Huband et al. recommend k = 4 for two objectives and k = 2 * (m - 1) for m objectives.
            kwargs: Arbitrary keyword arguments, please refer to the optproblems documentation for more details.

        Please refer to the optproblems documentation for the details of the available properties.
        https://www.simonwessing.de/optproblems/doc/wfg.html
        """
        assert 1 <= function_id <= 9, "function_id must be in [1, 9]"
        self._problem = wfg.WFG(num_objectives, num_variables, k, **kwargs)[function_id - 1]

        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(0.0, 2.0 * (i + 1))
            for i in range(num_variables)
        }

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE for _ in range(self._problem.num_objectives)]

    def evaluate(self, params: dict[str, float]) -> float:
        """Evaluate the objective function.
        Args:
            params:
                Decision variable, e.g., evaluate({"x0": 1.0, "x1": 2.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The objective value.

        """
        return self._problem.objective_function([params[name] for name in self._search_space])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)
