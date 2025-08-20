from __future__ import annotations

import math
from typing import Any

import optuna
import optunahub


try:
    import optproblems.dtlz
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install optproblems diversipy` to use `dtlz`.")


class Problem(optunahub.benchmarks.ConstrainedMixin, optunahub.benchmarks.BaseProblem):
    """Wrapper class for the DTLZ test suite of optproblems."""

    available_combinations = [
        {"constraint_type": 1, "function_id": 1},  # C1-DTLZ1
        {"constraint_type": 1, "function_id": 3},  # C1-DTLZ3
        {"constraint_type": 2, "function_id": 2},  # C2-DTLZ2
        {"constraint_type": 3, "function_id": 1},  # C3-DTLZ1
        {"constraint_type": 3, "function_id": 4},  # C3-DTLZ4
    ]

    def __init__(
        self,
        function_id: int,
        n_objectives: int,
        constraint_type: int,
        dimension: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the problem.
        Args:
            function_id: Function ID of the DTLZ problem in [1, 4].
            n_objectives: Number of objectives.
            dimension: Number of variables.
            kwargs: Arbitrary keyword arguments, please refer to the optproblems documentation for more details.

        Please refer to the optproblems documentation for the details of the available properties.
        https://www.simonwessing.de/optproblems/doc/dtlz.html
        """
        assert 1 <= function_id <= 4, "function_id must be in [1, 4]"
        if dimension is None:
            dimension = n_objectives + (4 if function_id in [1, 4] else 9)
        self._dtlz_type = {"constraint_type": constraint_type, "function_id": function_id}

        assert (
            self._dtlz_type in self.available_combinations
        ), f"Invalid combination of constraint_type and function_id: {self._dtlz_type}. Available combinations are: {self.available_combinations}"
        self._problem = optproblems.dtlz.DTLZ(n_objectives, dimension, **kwargs)[function_id - 1]

        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(
                self._problem.min_bounds[i], self._problem.max_bounds[i]
            )
            for i in range(dimension)
        }

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        direction = (
            optuna.study.StudyDirection.MAXIMIZE
            if self._problem.do_maximize
            else optuna.study.StudyDirection.MINIMIZE
        )
        return [direction] * self._problem.num_objectives

    def evaluate(self, params: dict[str, float]) -> list[float]:
        """Evaluate the objective functions.
        Args:
            params:
                Decision variable, e.g., evaluate({"x0": 0.0, "x1": 1.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The objective values.

        """
        return self._problem.objective_function([params[name] for name in self._search_space])

    def evaluate_constraints(self, params: dict[str, float]) -> list[float]:
        """Evaluate the constraint functions.
        Args:
            params:
                Decision variable, e.g., evaluate_constraints({"x0": 1.0, "x1": 2.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The constraint functions values.

        """
        objective_values = self.evaluate(params)
        if self._dtlz_type == {"constraint_type": 1, "function_id": 1}:
            return [objective_values[-1] / 0.6 + sum(objective_values[:-1]) / 0.5 - 1.0]
        elif self._dtlz_type == {"constraint_type": 1, "function_id": 3}:
            sum_squares = sum(x**2 for x in objective_values)
            r = 9.0 if (m := len(objective_values)) < 5 else 12.5 if m < 10 else 15.0
            return [-(sum_squares - 16) * (sum_squares - r**2)]
        elif self._dtlz_type == {"constraint_type": 2, "function_id": 2}:
            sum_squares = sum(x**2 for x in objective_values)
            m = len(objective_values)
            r = 0.3 if m == 3 else 0.5
            return [
                -max(
                    sum_squares - r**2 + 1.0 - 2.0 * max(objective_values),
                    sum((v - 1 / math.sqrt(m)) ** 2 for v in objective_values) - r**2,
                )
            ]
        elif self._dtlz_type == {"constraint_type": 3, "function_id": 1}:
            sum_values = sum(objective_values)
            return [-sum_values - v + 1.0 for v in objective_values]
        elif self._dtlz_type == {"constraint_type": 3, "function_id": 4}:
            squares = [x**2 for x in objective_values]
            return [-sum(squares) + v * 0.75 + 1.0 for v in objective_values]
        else:
            raise ValueError(f"Unsupported DTLZ type: {self._dtlz_type}")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)
