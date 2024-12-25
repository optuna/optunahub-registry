from __future__ import annotations

from typing import Any

import cocoex as ex
import optuna
import optunahub


class Problem(optunahub.benchmarks.BaseProblem):
    """Wrapper class for COCO bbob-biobj-mixint test suite.
    https://numbbo.github.io/coco/testsuites/bbob-biobj-mixint
    """

    def __init__(self, function_id: int, dimension: int, instance_id: int = 1):
        """Initialize the problem.
        Args:
            function_id: Function index in [1, 92].
            dimension: Dimension of the problem in [5, 10, 20, 40, 80, 160].
            instance_id: Instance index in [1, 15].

        Please refer to the COCO documentation for the details of the available properties.
        https://numbbo.github.io/coco-doc/apidocs/cocoex/cocoex.Problem.html
        """

        assert 1 <= function_id <= 92, "function_id must be in [1, 92]"
        possible_dimensions = [5, 10, 20, 40, 80, 160]
        assert dimension in possible_dimensions, f"dimension must be in {possible_dimensions}."
        assert 1 <= instance_id <= 15, "instance_id must be in [1, 15]"

        self._problem = ex.Suite(
            "bbob-biobj-mixint", "", ""
        ).get_problem_by_function_dimension_instance(
            function=function_id, dimension=dimension, instance=instance_id
        )

        # In bbob-biobj-mixint, first k variables are integer, and the rest are float.
        self._search_space = {}
        for i in range(dimension):
            dist_cls = (
                optuna.distributions.IntDistribution
                if i < self._problem.number_of_integer_variables
                else optuna.distributions.FloatDistribution
            )
            self._search_space[f"x{i}"] = dist_cls(
                low=self._problem.lower_bounds[i],
                high=self._problem.upper_bounds[i],
            )

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE, optuna.study.StudyDirection.MINIMIZE]

    def evaluate(self, params: dict[str, float]) -> tuple[float, float]:
        """Evaluate the objective function.
        Args:
            params:
                Decision variable, e.g., evaluate({"x0": 1.0, "x1": 2.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The objective value.

        """
        return tuple(self._problem([params[name] for name in self._search_space]))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def __del__(self) -> None:
        self._problem.free()
