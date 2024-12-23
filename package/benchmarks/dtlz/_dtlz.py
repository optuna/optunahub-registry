from __future__ import annotations

from typing import Any

import optproblems.dtlz
import optuna
import optunahub


class Problem(optunahub.benchmarks.BaseProblem):
    """Wrapper class for the DTLZ test suite of optproblems."""

    def __init__(
        self, function_id: int, num_objectives: int, num_variables: int, **kwargs: Any
    ) -> None:
        """Initialize the problem.
        Args:
            function_id: Function ID of the WFG problem in [1, 7].
            num_objectives: Number of objectives.
            num_variables: Number of variables.
            kwargs: Arbitrary keyword arguments, please refer to the optproblems documentation for more details.

        Please refer to the optproblems documentation for the details of the available properties.
        https://www.simonwessing.de/optproblems/doc/dtlz.html
        """
        assert 1 <= function_id <= 7, "function_id must be in [1, 7]"
        self._problem = {
            1: optproblems.dtlz.DTLZ1,
            2: optproblems.dtlz.DTLZ2,
            3: optproblems.dtlz.DTLZ3,
            4: optproblems.dtlz.DTLZ4,
            5: optproblems.dtlz.DTLZ5,
            6: optproblems.dtlz.DTLZ6,
            7: optproblems.dtlz.DTLZ7,
        }[function_id](num_objectives, num_variables, **kwargs)

        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(0.0, 1.0) for i in range(num_variables)
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
                Decision variable, e.g., evaluate({"x0": 0.0, "x1": 1.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The objective value.

        """
        return self._problem.objective_function([params[name] for name in self._search_space])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)
