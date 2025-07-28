from __future__ import annotations

from typing import Any

import optuna
import optunahub


try:
    import optproblems.wfg
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install optproblems diversipy` to use `wfg`.")


class Problem(optunahub.benchmarks.BaseProblem):
    """Wrapper class for the WFG test suite of optproblems."""

    def __init__(
        self,
        function_id: int,
        n_objectives: int,
        dimension: int,
        k: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the problem.
        Args:
            function_id: Function ID of the WFG problem in [1, 9].
            n_objectives: Number of objectives.
            dimension: Number of variables.
            k: Number of position parameters. It must hold k < dimension and k must be a multiple of n_objectives - 1. Huband et al. recommend k = 4 for two objectives and k = 2 * (m - 1) for m objectives.
            kwargs: Arbitrary keyword arguments, please refer to the optproblems documentation for more details.

        Please refer to the optproblems documentation for the details of the available properties.
        https://www.simonwessing.de/optproblems/doc/wfg.html
        """
        assert 1 <= function_id <= 9, "function_id must be in [1, 9]"

        if k is None:
            k = 2 * (n_objectives - 1) if n_objectives > 2 else 4

        self._problem = optproblems.wfg.WFG(n_objectives, dimension, k, **kwargs)[function_id - 1]

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

    @property
    def reference_point(self) -> list[float]:
        """Return the commonly-used reference point for the problem."""
        return [float(3 + 2 * i) for i in range(self._problem.num_objectives)]

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
