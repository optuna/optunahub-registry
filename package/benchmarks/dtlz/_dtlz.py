from __future__ import annotations

from typing import Any

import optuna
import optunahub


try:
    import optproblems.dtlz
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install optproblems diversipy` to use `dtlz`.")


class Problem(optunahub.benchmarks.BaseProblem):
    """Wrapper class for the DTLZ test suite of optproblems."""

    def __init__(self, function_id: int, n_objectives: int, dimension: int, **kwargs: Any) -> None:
        """Initialize the problem.
        Args:
            function_id: Function ID of the DTLZ problem in [1, 7].
            n_objectives: Number of objectives.
            dimension: Number of variables.
            kwargs: Arbitrary keyword arguments, please refer to the optproblems documentation for more details.

        Please refer to the optproblems documentation for the details of the available properties.
        https://www.simonwessing.de/optproblems/doc/dtlz.html
        """
        assert 1 <= function_id <= 7, "function_id must be in [1, 7]"
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

    def evaluate(self, params: dict[str, float]) -> float:
        """Evaluate the objective functions.
        Args:
            params:
                Decision variable, e.g., evaluate({"x0": 0.0, "x1": 1.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The objective values.

        """
        return self._problem.objective_function([params[name] for name in self._search_space])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)
