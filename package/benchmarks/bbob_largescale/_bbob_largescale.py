from __future__ import annotations

from typing import Any

import optuna
import optunahub


try:
    import cocoex as ex
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install coco-experiment` to use `bbob_largescale`.")


class Problem(optunahub.benchmarks.BaseProblem):
    """Wrapper class for COCO bbob-largescale test suite.
    https://numbbo.github.io/coco/testsuites/bbob-largescale

    1 Separable Functions
        f1: Sphere Function
        f2: Separable Ellipsoidal Function
        f3: Rastrigin Function
        f4: Büche-Rastrigin Function
        f5: Linear Slope
    2 Functions with low or moderate conditioning
        f6: Attractive Sector Function
        f7: Step Ellipsoidal Function
        f8: Rosenbrock Function, original
        f9: Rosenbrock Function, rotated
    3 Functions with high conditioning and unimodal
        f10: Ellipsoidal Function
        f11: Discus Function
        f12: Bent Cigar Function
        f13: Sharp Ridge Function
        f14: Different Powers Function
    4 Multi-modal functions with adequate global structure
        f15: Rastrigin Function
        f16: Weierstrass Function
        f17: Schaffer's F7 Function
        f18: Schaffer's F7 Function, moderately ill-conditioned
        f19: Composite Griewank-Rosenbrock Function F8F2
    5 Multi-modal functions with weak global structure
        f20: Schwefel Function
        f21: Gallagher's Gaussian 101-me Peaks Function
        f22: Gallagher's Gaussian 21-hi Peaks Function
        f23: Katsuura Function
        f24: Lunacek bi-Rastrigin Function
    """

    def __init__(self, function_id: int, dimension: int, instance_id: int = 1):
        """Initialize the problem.
        Args:
            function_id: Function index in [1, 24].
            dimension: Dimension of the problem in [20, 40, 80, 160, 320, 640].
            instance_id: Instance index in [1, 15].

        Please refer to the COCO documentation for the details of the available properties.
        https://numbbo.github.io/coco-doc/apidocs/cocoex/cocoex.Problem.html
        """

        self._valid_arguments = False
        assert 1 <= function_id <= 24, "function_id must be in [1, 24]"
        possible_dimensions = [20, 40, 80, 160, 320, 640]
        assert dimension in possible_dimensions, f"dimension must be in {possible_dimensions}."
        assert 1 <= instance_id <= 15, "instance_id must be in [1, 15]"
        self._valid_arguments = True

        self._problem = ex.Suite(
            "bbob-largescale", "", ""
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
        return [optuna.study.StudyDirection.MINIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        """Evaluate the objective function.
        Args:
            params:
                Decision variable, e.g., evaluate({"x0": 1.0, "x1": 2.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The objective value.

        """
        return self._problem([params[name] for name in self._search_space])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def __del__(self) -> None:
        if not self._valid_arguments:
            return

        self._problem.free()
