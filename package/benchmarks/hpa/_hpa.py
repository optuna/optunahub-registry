from __future__ import annotations

from typing import Any
from typing import Protocol
from typing import TYPE_CHECKING

import numpy as np
import optuna
import optunahub

from .hpa_original import problem


class BaseUnconstrainedHPABenchmark(Protocol):
    nf: int  # Number of objectives
    nx: int  # Number of parameters
    ng: int  # Number of constraints (always 0 for unconstrained problems)

    def __init__(self, n_div: int, level: int) -> None:
        raise NotImplementedError

    def __call__(self, x: list[float]) -> np.ndarray:
        raise NotImplementedError


class BaseConstrainedHPABenchmark(Protocol):
    nf: int  # Number of objectives
    nx: int  # Number of parameters
    ng: int  # Number of constraints (always positive)

    def __init__(self, n_div: int, level: int) -> None:
        raise NotImplementedError

    def __call__(self, x: list[float]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


hpa_benchmark_names = [
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

constrained_problems: dict[str, type[BaseConstrainedHPABenchmark]] = {
    name: getattr(problem, name) for name in hpa_benchmark_names if name[-2] != "0"
}

unconstrained_problems: dict[str, type[BaseUnconstrainedHPABenchmark]] = {
    name: getattr(problem, name) for name in hpa_benchmark_names if name[-2] == "0"
}


class Problem(optunahub.benchmarks.BaseProblem):
    def __init__(self, problem_name: str, n_div: int = 4, level: int = 0) -> None:
        """Initialize the problem.
        Args:
            problem_name: Name of problem.
            n_div: Number of the wing segmentation.
            level: Number of the difficulty level of the problem in [0, 2].

        Please refer to the hpa repository for the details.
        https://github.com/Nobuo-Namura/hpa
        """
        assert (
            problem_name in unconstrained_problems
        ), f"problem_name must be in {list(unconstrained_problems.keys())}"
        assert n_div > 0 and isinstance(n_div, int), "n_div must be an integer greater than 0"
        assert level in [0, 1, 2], "level must be in [0, 1, 2]"

        self.problem_name = problem_name

        self._problem = unconstrained_problems[problem_name](n_div=n_div, level=level)

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

    def evaluate(self, params: dict[str, float]) -> list[float]:
        return self._problem([params[name] for name in self._search_space]).tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)


class ConstrainedProblem(optunahub.benchmarks.ConstrainedMixin, optunahub.benchmarks.BaseProblem):
    def __init__(self, problem_name: str, n_div: int = 4, level: int = 0) -> None:
        """Initialize the problem.
        Args:
            problem_name: Name of problem.
            n_div: Number of the wing segmentation.
            level: Number of the difficulty level of the problem in [0, 2].

        Please refer to the hpa repository for the details.
        https://github.com/Nobuo-Namura/hpa
        """
        if problem_name not in constrained_problems:
            raise ValueError(
                f"problem_name must be in {list(constrained_problems.keys())}, "
                f"but got {problem_name}."
            )

        if n_div <= 0 or not isinstance(n_div, int):
            raise ValueError(f"n_div must be an positive integer, but got {n_div}.")

        if level not in [0, 1, 2]:
            raise ValueError(f"level must be in [0, 1, 2], but got {level}")

        self.problem_name = problem_name

        self._problem = constrained_problems[problem_name](n_div=n_div, level=level)

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

    def evaluate(self, params: dict[str, float]) -> list[float]:
        return self._problem([params[name] for name in self._search_space])[0].tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def evaluate_constraints(self, params: dict[str, float]) -> list[float]:
        return self._problem([params[name] for name in self._search_space])[1].tolist()


if TYPE_CHECKING:
    for problem_cls in constrained_problems.values():
        _0: BaseConstrainedHPABenchmark = problem_cls(n_div=4, level=0)
    for problem_cls in unconstrained_problems.values():
        _1: BaseUnconstrainedHPABenchmark = problem_cls(n_div=4, level=0)
