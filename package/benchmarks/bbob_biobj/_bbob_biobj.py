from __future__ import annotations

from typing import Any

import optuna
import optunahub


try:
    import cocoex as ex
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install coco-experiment` to use `bbob_biobj`.")


class Problem(optunahub.benchmarks.BaseProblem):
    """Wrapper class for COCO bbob-biobj/bbob-biobj-ext test suite.
    https://coco-platform.org/testsuites/bbob-biobj/def.html
    """

    def __init__(self, function_id: int, dimension: int, instance_id: int = 1):
        """Initialize the problem.
        Args:
            function_id: Function index in [1, 92].
            dimension: Dimension of the problem in [2, 3, 5, 10, 20, 40].
            instance_id: Instance index in [1, 15].

        Please refer to the COCO documentation for the details of the available properties.
        https://coco-platform.org/testsuites/bbob-biobj/overview.html
        """

        assert 1 <= function_id <= 92, "function_id must be in [1, 92]"
        assert dimension in [2, 3, 5, 10, 20, 40], "dimension must be in [2, 3, 5, 10, 20, 40]"
        assert 1 <= instance_id <= 15, "instance_id must be in [1, 15]"

        # The first 55 functions of the bbob-biobj-ext suite are the same as in the original bbob-biobj test suite
        # to which 37 functions are added. So we always use the bbob-biobj-ext suite for simplicity.
        self._problem = ex.Suite(
            "bbob-biobj-ext", "", ""
        ).get_problem_by_function_dimension_instance(
            function=function_id, dimension=dimension, instance=instance_id
        )
        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(
                low=self._problem.lower_bounds[i],
                high=self._problem.upper_bounds[i],
            )
            for i in range(self._problem.dimension)
        }

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
