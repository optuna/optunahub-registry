from typing import Any
from typing import List

import optuna
import optunahub

import hpa.problem


_problem_names = [
    "HPA131",
    "HPA101",
    "HPA142",
    "HPA102",
    "HPA143",
    "HPA103",
    "HPA241",
    "HPA201",
    "HPA222",
    "HPA202",
    "HPA233",
    "HPA203",
    "HPA244",
    "HPA204",
    "HPA245",
    "HPA205",
    "HPA341",
    "HPA301",
    "HPA322",
    "HPA302",
    "HPA333",
    "HPA303",
    "HPA344",
    "HPA304",
    "HPA345",
    "HPA305",
    "HPA441",
    "HPA401",
    "HPA422",
    "HPA402",
    "HPA443",
    "HPA403",
    "HPA541",
    "HPA501",
    "HPA542",
    "HPA502",
    "HPA641",
    "HPA601",
    "HPA941",
    "HPA901",
]

_constrained_problem_names = [
    "HPA131",
    "HPA142",
    "HPA143",
    "HPA241",
    "HPA222",
    "HPA233",
    "HPA244",
    "HPA245",
    "HPA341",
    "HPA322",
    "HPA333",
    "HPA344",
    "HPA345",
    "HPA441",
    "HPA422",
    "HPA443",
    "HPA541",
    "HPA542",
    "HPA641",
    "HPA941",
]


class Problem(optunahub.benchmarks.BaseProblem):
    def __init__(
        self, problem_name: str, n_div: int = 4, level: int = 0, NORMALIZED: bool = True
    ) -> None:
        """Initialize the problem.
        Args:
            problem_name: Name of problem.
            n_div: Number of the wing segmentation.
            level: Number of the difficulty level of the problem in [0, 2].
            NORMALIZED: A boolean indicating use of normalized design variables (True) or not (False).

        Please refer to the hpa repository for the details.
        https://github.com/Nobuo-Namura/hpa
        """
        assert problem_name in _problem_names, f"problem_name must be in {_problem_names}"
        assert n_div > 0 and isinstance(n_div, int), "n_div must be an integer greater than 0"
        assert level in [0, 1, 2], "level must be in [0, 1, 2]"
        assert isinstance(NORMALIZED, bool), "NORMALIZED must be a boolean"

        self._problem_name = problem_name
        self._n_div = n_div
        self._level = level
        self._normalized = NORMALIZED

        self._problem = getattr(hpa.problem, problem_name)(
            n_div=self._n_div,
            level=self._level,
            NORMALIZED=self._normalized,
        )

        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(0, 1) for i in range(self._problem.nx)
        }

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE] * self._problem.nf

    def evaluate(self, params: dict[str, float]) -> List[float]:
        if self._problem_name in _constrained_problem_names:
            return self._problem([params[name] for name in self._search_space])[0].tolist()
        return self._problem([params[name] for name in self._search_space]).tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)


class ConstrainedProblem(optunahub.benchmarks.ConstrainedMixin, Problem):
    def evaluate_constraints(self, params: dict[str, float]) -> List[float]:
        if self._problem_name in _constrained_problem_names:
            return self._problem([params[name] for name in self._search_space])[1].tolist()
        else:
            raise TypeError(f"{self._problem_name} is not a constrained problem.")
