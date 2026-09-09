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

# The 11 fundamental objectives and the 5 constraints of Table 1 in the paper
# (https://arxiv.org/abs/2312.08953). The keys are the indices of `f_i` and `g_j` in the paper.
# `negative_` is prefixed to the quantities that the paper maximizes, because the original
# implementation negates them so that every objective is minimized.
_OBJECTIVE_NAMES: dict[int, str] = {
    1: "f1_required_power",  # P [W]
    2: "f2_drag",  # D [N]
    3: "f3_negative_cruise_speed",  # -V [m/s]
    4: "f4_max_wingtip_deflection",  # max(|delta|, |delta_park|) [m]
    5: "f5_max_twist_angle",  # Phi [deg]
    6: "f6_negative_wing_efficiency",  # -E [-]
    7: "f7_empty_weight",  # W_0 [kg]
    8: "f8_wing_span",  # B [m]
    9: "f9_root_angle_of_attack",  # alpha_0 [deg]
    10: "f10_wire_tension",  # T [N]
    11: "f11_negative_payload",  # -W_p [kg]
}
_CONSTRAINT_NAMES: dict[int, str] = {
    1: "g1_max_strain",  # n_m * n_s * eps_max / eps_u - 1 [-]
    2: "g2_wingtip_dihedral_angle",  # B * (sin(gamma) - sin(gamma_u)) / 2 [m]
    3: "g3_parked_wingtip_deflection",  # -delta_park [m]
    4: "g4_required_power",  # P - P_max [W]
    5: "g5_min_cruise_speed",  # 1 - (V / V_min)^3 [-]
}

# Table 2 in the paper lists the objectives of a constrained problem and of its unconstrained
# counterpart in the same row, e.g. HPA131 and HPA101, so each row below names both problems.
# The indices follow the order in which the original implementation returns the objectives.
_OBJECTIVE_INDICES_TABLE: list[tuple[tuple[int, ...], tuple[str, ...]]] = [
    ((2,), ("HPA131", "HPA101")),
    ((1,), ("HPA142", "HPA102")),
    ((3,), ("HPA143", "HPA103")),
    ((1, 3), ("HPA241", "HPA201")),
    ((1, 4), ("HPA222", "HPA202")),
    ((2, 11), ("HPA233", "HPA203")),
    ((3, 5), ("HPA244", "HPA204")),
    ((3, 10), ("HPA245", "HPA205")),
    ((1, 3, 6), ("HPA341", "HPA301")),
    ((1, 4, 5), ("HPA322", "HPA302")),
    ((2, 9, 11), ("HPA333", "HPA303")),
    ((2, 3, 5), ("HPA344", "HPA304")),
    ((1, 3, 10), ("HPA345", "HPA305")),
    ((2, 3, 5, 7), ("HPA441", "HPA401")),
    ((1, 4, 5, 8), ("HPA422", "HPA402")),
    ((3, 6, 7, 10), ("HPA443", "HPA403")),
    ((2, 3, 5, 6, 7), ("HPA541", "HPA501")),
    ((1, 3, 9, 10, 11), ("HPA542", "HPA502")),
    ((1, 3, 5, 6, 7, 8), ("HPA641", "HPA601")),
    ((1, 3, 5, 6, 7, 8, 9, 10, 11), ("HPA941", "HPA901")),
]

# Constrained problems share only four distinct constraint sets. The indices follow the order in
# which the original implementation returns the constraints, which is not sorted by the index.
_CONSTRAINT_INDICES_TABLE: list[tuple[tuple[int, ...], tuple[str, ...]]] = [
    ((1, 3, 2), ("HPA131", "HPA233", "HPA333")),
    ((1, 5), ("HPA222", "HPA322", "HPA422")),
    ((1, 3, 2, 5), ("HPA142",)),
    (
        (1, 3, 2, 4),
        (
            "HPA143",
            "HPA241",
            "HPA244",
            "HPA245",
            "HPA341",
            "HPA344",
            "HPA345",
            "HPA441",
            "HPA443",
            "HPA541",
            "HPA542",
            "HPA641",
            "HPA941",
        ),
    ),
]

_OBJECTIVE_INDICES: dict[str, tuple[int, ...]] = {
    problem_name: indices
    for indices, problem_names in _OBJECTIVE_INDICES_TABLE
    for problem_name in problem_names
}
_CONSTRAINT_INDICES: dict[str, tuple[int, ...]] = {
    problem_name: indices
    for indices, problem_names in _CONSTRAINT_INDICES_TABLE
    for problem_name in problem_names
}


def _build_metric_names(problem_name: str, n_objectives: int) -> list[str]:
    names = [_OBJECTIVE_NAMES[i] for i in _OBJECTIVE_INDICES[problem_name]]
    assert len(names) == n_objectives, f"{problem_name} must have {n_objectives} objectives."
    return names


def _build_constraint_names(problem_name: str, n_constraints: int) -> list[str]:
    names = [_CONSTRAINT_NAMES[i] for i in _CONSTRAINT_INDICES[problem_name]]
    assert len(names) == n_constraints, f"{problem_name} must have {n_constraints} constraints."
    return names


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
        if problem_name not in unconstrained_problems:
            raise ValueError(
                f"problem_name must be in {list(unconstrained_problems.keys())}, "
                f"but got {problem_name}."
            )

        if n_div <= 0 or not isinstance(n_div, int):
            raise ValueError(f"n_div must be an positive integer, but got {n_div}.")

        if level not in [0, 1, 2]:
            raise ValueError(f"level must be in [0, 1, 2], but got {level}")

        self.problem_name = problem_name

        self._problem = unconstrained_problems[problem_name](n_div=n_div, level=level)

        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(0, 1) for i in range(self._problem.nx)
        }
        self._metric_names = _build_metric_names(problem_name, self._problem.nf)

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE] * self._problem.nf

    @property
    def metric_names(self) -> list[str]:
        """Return the objective names in the order returned by ``evaluate``."""
        return self._metric_names.copy()

    def evaluate(self, params: dict[str, float]) -> list[float]:
        return self._problem([params[name] for name in self._search_space]).tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)


class ConstrainedProblem(optunahub.benchmarks.BaseProblem):
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
        self._metric_names = _build_metric_names(problem_name, self._problem.nf)
        self._constraint_names = _build_constraint_names(problem_name, self._problem.ng)

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE] * self._problem.nf

    @property
    def metric_names(self) -> list[str]:
        """Return the objective names in the order returned by ``evaluate``."""
        return self._metric_names.copy()

    @property
    def constraint_names(self) -> list[str]:
        """Return the constraint names used as the keys of ``evaluate_constraints``."""
        return self._constraint_names.copy()

    def evaluate(self, params: dict[str, float]) -> list[float]:
        return self._problem([params[name] for name in self._search_space])[0].tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def evaluate_constraints(self, params: dict[str, float]) -> dict[str, float]:
        values = self._problem([params[name] for name in self._search_space])[1].tolist()
        return dict(zip(self._constraint_names, values))


if TYPE_CHECKING:
    for problem_cls in constrained_problems.values():
        _0: BaseConstrainedHPABenchmark = problem_cls(n_div=4, level=0)
    for problem_cls in unconstrained_problems.values():
        _1: BaseUnconstrainedHPABenchmark = problem_cls(n_div=4, level=0)
