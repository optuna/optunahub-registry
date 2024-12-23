from __future__ import annotations

from typing import Any

import optproblems.zdt
import optuna
import optunahub


class Problem(optunahub.benchmarks.BaseProblem):
    """Wrapper class for the WFG test suite of optproblems."""

    def __init__(self, function_id: int, **kwargs: Any) -> None:
        """Initialize the problem.
        Args:
            function_id: Function ID of the WFG problem in [1, 6].
            kwargs: Arbitrary keyword arguments, please refer to the optproblems documentation for more details.

        Please refer to the optproblems documentation for the details of the available properties.
        https://www.simonwessing.de/optproblems/doc/zdt.html
        """
        assert 1 <= function_id <= 6, "function_id must be in [1, 6]"
        self._problem = {
            1: optproblems.zdt.ZDT1,
            2: optproblems.zdt.ZDT2,
            3: optproblems.zdt.ZDT3,
            4: optproblems.zdt.ZDT4,
            5: optproblems.zdt.ZDT5,
            6: optproblems.zdt.ZDT6,
        }[function_id](**kwargs)

        if function_id in [1, 2, 3]:
            self._search_space = {
                f"x{i}": optuna.distributions.FloatDistribution(0.0, 1.0) for i in range(30)
            }
        elif function_id in [4, 6]:
            self._search_space = {
                f"x{i}": optuna.distributions.FloatDistribution(0.0, 1.0) for i in range(10)
            }
        else:
            # x0 \in {0, 1}^{30}
            self._search_space = {"x0": optuna.distributions.IntDistribution(0, 1 << 29)}
            # x1, ..., x11 \in {0, 1}^{5}
            for i in range(1, 11):
                self._search_space[f"x{i}"] = optuna.distributions.IntDistribution(0, 1 << 4)

        self._function_id = function_id

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE, optuna.study.StudyDirection.MINIMIZE]

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
        phenome = []
        for name in self.search_space:
            x = params[name]
            assert isinstance(x, int), f"{name} must be an integer"
            if name == "x0":
                # e.g., 0b101011 -> [0, ..., 0, 1, 0, 1, 0, 1, 1]
                bitvector = [x >> i & 1 for i in range(30)][::-1]
            else:
                bitvector = [x >> i & 1 for i in range(5)][::-1]
            phenome.append(bitvector)
        return self._problem.objective_function(phenome)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)
