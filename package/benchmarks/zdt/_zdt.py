from __future__ import annotations

from typing import Any

import optproblems.zdt
import optuna
import optunahub


class Problem(optunahub.benchmarks.BaseProblem):
    """Wrapper class for the ZDT test suite of optproblems."""

    def __init__(self, function_id: int, **kwargs: Any) -> None:
        """Initialize the problem.
        Args:
            function_id: Function ID of the ZDT problem in [1, 6].
            kwargs: Arbitrary keyword arguments, please refer to the optproblems documentation for more details.

        Please refer to the optproblems documentation for the details of the available properties.
        https://www.simonwessing.de/optproblems/doc/zdt.html
        """
        assert 1 <= function_id <= 6, "function_id must be in [1, 6]"
        self._problem = optproblems.zdt.ZDT(**kwargs)[function_id - 1]

        if function_id != 5:
            self._search_space = {
                f"x{i}": optuna.distributions.FloatDistribution(
                    self._problem.min_bounds[i], self._problem.max_bounds[i]
                ) for i in range(num_variables)
            }
        else:
            self._search_space = {}
            for i, binary_length in enumerate([30] + [5]*10):
                self._search_space.update(
                    {
                        f"x{i}_{b}": optuna.distributions.CategoricalDistribution([True, False])
                        for b in range(binary_length)
                    }
                )

        self._function_id = function_id

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
        """Evaluate the objective function.
        Args:
            params:
                Decision variable, e.g., evaluate({"x0": 0.0, "x1": 1.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The objective value.

        """
        if self._function_id != 5:
            return self._problem.objective_function([params[name] for name in self._search_space])

        # ZDT5 is a special case
        binary_lengths = [30] + [5]*10
        phenome = [
            [int(params[f"x{i}_{b}"]) for b in range(binary_length)]
            for i, binary_length in enumerate(binary_lengths)        
        ]
        return self._problem.objective_function(phenome)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)
