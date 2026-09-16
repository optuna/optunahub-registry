from __future__ import annotations

from typing import Any
from typing import NamedTuple
from typing import Protocol
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna.distributions import FloatDistribution
import optunahub

from .reproblem_original import problem


class BaseUnconstrainedREBenchmark(Protocol):
    problem_name: str
    n_objectives: int  # Number of objectives
    n_variables: int  # Number of parameters
    n_original_constraints: int  # Number of constraints folded into the objectives
    lbound: np.ndarray  # Lower bound of each parameter
    ubound: np.ndarray  # Upper bound of each parameter

    def __init__(self) -> None:
        raise NotImplementedError

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BaseConstrainedREBenchmark(Protocol):
    problem_name: str
    n_objectives: int  # Number of objectives
    n_variables: int  # Number of parameters
    n_constraints: int  # Number of constraints (always positive)
    lbound: np.ndarray  # Lower bound of each parameter
    ubound: np.ndarray  # Upper bound of each parameter

    def __init__(self) -> None:
        raise NotImplementedError

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


unconstrained_problem_names = [
    "RE21",
    "RE22",
    "RE23",
    "RE24",
    "RE25",
    "RE31",
    "RE32",
    "RE33",
    "RE34",
    "RE35",
    "RE36",
    "RE37",
    "RE41",
    "RE42",
    "RE61",
    "RE91",
]

constrained_problem_names = [
    "CRE12",
    "CRE13",
    "CRE14",
    "CRE15",
    "CRE21",
    "CRE22",
    "CRE23",
    "CRE24",
    "CRE25",
    "CRE31",
    "CRE32",
    "CRE51",
]

unconstrained_problems: dict[str, type[BaseUnconstrainedREBenchmark]] = {
    name: getattr(problem, name) for name in unconstrained_problem_names
}

constrained_problems: dict[str, type[BaseConstrainedREBenchmark]] = {
    name: getattr(problem, name) for name in constrained_problem_names
}


# The objective of an unconstrained problem that aggregates the constraints folded into it.
_TOTAL_CONSTRAINT_VIOLATION = "total_constraint_violation"


class _ProblemInfo(NamedTuple):
    original_name: str  # Name of the original problem, as in Table 1 of the paper.
    unconstrained_name: str
    constrained_name: str | None  # `None` when the suite has no constrained counterpart.
    objective_names: tuple[str, ...]  # The original objectives shared by both problems.
    constraint_names: tuple[str, ...]  # The constraints in the order `evaluate` returns them.


# Table 1 of the paper (https://arxiv.org/abs/2009.12867) derives an unconstrained problem and its
# constrained counterpart from the same original engineering problem, e.g. both RE31 and CRE21 are
# the two bar truss design problem, so each row below covers such a pair at once. A pair shares the
# original problem name and the original objectives, and the unconstrained problem additionally
# folds the constraints into an aggregated violation objective, while the constrained problem
# exposes them as constraints.
# CRE11-CRE14 are not part of the paper's Table 1: RE22-RE25 already fold real constraints into their
# violation objective, so this package exposes those same constraints as their constrained
# counterpart, following the naming convention of the pairs the paper does define.
# The objective names carry the `f_i` index used by the problem definitions in the supplementary
# file (https://github.com/ryojitanabe/reproblems/blob/master/doc/re-supplementary_file.pdf), and
# `negative_` is prefixed to the quantities that the paper maximizes, because the original
# implementation negates them so that every objective is minimized.
# The constraint names carry the `g_j` index the same way and are listed in the order the original
# implementation returns them, which is not sorted by `j` for the conceptual marine design problem.
# Neither the paper nor the supplementary file names a constraint, so the descriptive part of each
# name is read off the constraint formula together with the original reference that the paper cites
# for that problem. The water resource planning problem is the one exception: its constraints are
# regression surrogates that bound quantities the reference leaves unnamed, so they keep the bare
# `g_j` index. The constraint names of a problem without a constrained counterpart document what its
# aggregated violation objective sums up, because only `ConstrainedProblem` exposes them as names.
_PROBLEM_INFO_TABLE: list[_ProblemInfo] = [
    _ProblemInfo(
        "FourBarTruss",
        "RE21",
        None,
        ("f1_structural_volume", "f2_joint_displacement"),
        (),
    ),
    _ProblemInfo(
        "ReinforcedConcreteBeam",
        "RE22",
        "CRE12",
        ("f1_total_cost",),
        ("g1_flexural_capacity", "g2_depth_to_width_ratio"),
    ),
    _ProblemInfo(
        "PressureVessel",
        "RE23",
        "CRE13",
        ("f1_total_cost",),
        ("g1_shell_thickness", "g2_head_thickness", "g3_working_volume"),
    ),
    _ProblemInfo(
        "HatchCover",
        "RE24",
        "CRE14",
        ("f1_weight",),
        ("g1_bending_stress", "g2_shear_stress", "g3_deflection", "g4_buckling_stress"),
    ),
    _ProblemInfo(
        "CoilCompressionSpring",
        "RE25",
        "CRE15",
        ("f1_volume",),
        (
            "g1_shear_stress",
            "g2_free_length",
            "g3_coil_to_wire_diameter_ratio",
            "g4_deflection_under_preload",
            "g5_combined_deflection_clearance",
            "g6_working_deflection",
        ),
    ),
    # `x1` and `x2` are the cross sectional areas of the members AC and BC, and `g2` and `g3` bound
    # the stress of the respective member by the same 1e5 kPa limit, while `g1` bounds `f1`.
    # `f2` is therefore the stress of the member AC rather than a joint displacement: it is the very
    # quantity `g2` bounds, and a displacement would not share a limit with the stress in `g3`.
    # Coello and Pulido (2005), which the paper cites for this problem, likewise gives the two
    # objectives as the volume of the truss and the stress of the member AC.
    _ProblemInfo(
        "TwoBarTruss",
        "RE31",
        "CRE21",
        ("f1_structural_weight", "f2_member_ac_stress"),
        ("g1_structural_weight", "g2_member_ac_stress", "g3_member_bc_stress"),
    ),
    _ProblemInfo(
        "WeldedBeam",
        "RE32",
        "CRE22",
        ("f1_cost", "f2_end_deflection"),
        (
            "g1_weld_shear_stress",
            "g2_beam_bending_stress",
            "g3_geometric_side",
            "g4_buckling_load",
        ),
    ),
    _ProblemInfo(
        "DiscBrake",
        "RE33",
        "CRE23",
        ("f1_brake_mass", "f2_minimum_stopping_time"),
        (
            "g1_minimum_radial_thickness",
            "g2_maximum_actuating_pressure",
            "g3_maximum_temperature",
            "g4_minimum_braking_torque",
        ),
    ),
    _ProblemInfo(
        "VehicleCrashworthiness",
        "RE34",
        None,
        ("f1_weight", "f2_acceleration_characteristics", "f3_toe_board_intrusion"),
        (),
    ),
    # `x1` is the gear face width, `x2` the tooth module, `x3` the number of teeth of the pinion,
    # `x4` and `x5` the shaft lengths between the bearings, and `x6` and `x7` the shaft diameters,
    # so `g5` to `g9` are the dimensional restrictions and `g10` and `g11` the shaft stresses.
    _ProblemInfo(
        "SpeedReducer",
        "RE35",
        "CRE24",
        ("f1_volume", "f2_gear_shaft_stress"),
        (
            "g1_gear_tooth_bending_stress",
            "g2_gear_tooth_contact_stress",
            "g3_shaft1_transverse_deflection",
            "g4_shaft2_transverse_deflection",
            "g5_pinion_pitch_diameter",
            "g6_maximum_face_width_to_module",
            "g7_minimum_face_width_to_module",
            "g8_shaft1_length_clearance",
            "g9_shaft2_length_clearance",
            "g10_shaft1_stress",
            "g11_shaft2_stress",
        ),
    ),
    _ProblemInfo(
        "GearTrain",
        "RE36",
        "CRE25",
        ("f1_gear_ratio_error", "f2_max_gear_size"),
        ("g1_gear_ratio_error",),
    ),
    _ProblemInfo(
        "RocketInjector",
        "RE37",
        None,
        (
            "f1_max_injector_face_temperature",
            "f2_inlet_distance",
            "f3_max_post_tip_temperature",
        ),
        (),
    ),
    _ProblemInfo(
        "CarSideImpact",
        "RE41",
        "CRE31",
        ("f1_car_weight", "f2_pubic_force", "f3_v_pillar_average_velocity"),
        (
            "g1_abdomen_load",
            "g2_upper_viscous_criterion",
            "g3_middle_viscous_criterion",
            "g4_lower_viscous_criterion",
            "g5_upper_rib_deflection",
            "g6_middle_rib_deflection",
            "g7_lower_rib_deflection",
            "g8_pubic_symphysis_force",
            "g9_b_pillar_velocity",
            "g10_front_door_velocity",
        ),
    ),
    # The original implementation of the conceptual marine design problem returns the seventh
    # constraint before the sixth one, so `constraint_names` is not sorted for this pair. The two
    # are the halves of the single two-sided bound `3000 <= DWT <= 500000` on the deadweight.
    _ProblemInfo(
        "ConceptualMarine",
        "RE42",
        "CRE32",
        (
            "f1_transportation_cost",
            "f2_light_ship_weight",
            "f3_negative_annual_cargo_transport_capacity",
        ),
        (
            "g1_length_to_beam_ratio",
            "g2_length_to_depth_ratio",
            "g3_length_to_draught_ratio",
            "g4_draught_to_deadweight",
            "g5_draught_to_depth",
            "g7_maximum_deadweight",
            "g6_minimum_deadweight",
            "g8_froude_number",
            "g9_metacentric_height",
        ),
    ),
    _ProblemInfo(
        "WaterResourcePlanning",
        "RE61",
        "CRE51",
        (
            "f1_drainage_network_cost",
            "f2_storage_facility_cost",
            "f3_treatment_facility_cost",
            "f4_expected_flood_damage_cost",
            "f5_expected_economic_loss_due_to_flood",
        ),
        ("g1", "g2", "g3", "g4", "g5", "g6", "g7"),
    ),
    # Unlike the other problems, the car cab design problem keeps each folded constraint as its own
    # objective instead of aggregating them, so its violation objectives are named individually.
    _ProblemInfo(
        "CarCab",
        "RE91",
        None,
        ("f1_car_weight",) + tuple(f"f{i + 1}_g{i}_violation" for i in range(1, 9)),
        (),
    ),
]

_ORIGINAL_NAMES: dict[str, str] = {
    problem_name: info.original_name
    for info in _PROBLEM_INFO_TABLE
    for problem_name in (info.unconstrained_name, info.constrained_name)
    if problem_name is not None
}

# The constraints that the unconstrained problem folds into its violation objective. They are not
# exposed as names, so this is only used to validate the table against the original implementation.
_FOLDED_CONSTRAINT_NAMES: dict[str, tuple[str, ...]] = {
    info.unconstrained_name: info.constraint_names for info in _PROBLEM_INFO_TABLE
}


def _build_name_tables() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Map every problem name to its objective names and, if constrained, its constraint names."""
    metric_names: dict[str, tuple[str, ...]] = {}
    constraint_names: dict[str, tuple[str, ...]] = {}
    for info in _PROBLEM_INFO_TABLE:
        objective_names = info.objective_names
        if info.constraint_names:
            index = len(objective_names) + 1
            objective_names += (f"f{index}_{_TOTAL_CONSTRAINT_VIOLATION}",)

        metric_names[info.unconstrained_name] = objective_names
        if info.constrained_name is None:
            continue

        # The constrained counterpart exposes the constraints instead of folding them in, so it
        # keeps only the original objectives.
        metric_names[info.constrained_name] = info.objective_names
        constraint_names[info.constrained_name] = info.constraint_names

    return metric_names, constraint_names


_METRIC_NAMES, _CONSTRAINT_NAMES = _build_name_tables()


def _modify_search_space_for_hard_problems(
    problem_name: str, search_space: dict[str, FloatDistribution], enable: bool
) -> dict[str, FloatDistribution]:
    if problem_name not in ["CRE21", "RE31"] or not enable:
        return search_space
    # NOTE(nabe): Special treatment. Without log-transform, random sampling cannot find feasible
    # solutions even with 10**6 trials.
    x0_dist = search_space["x0"]
    x1_dist = search_space["x1"]
    search_space["x0"] = FloatDistribution(x0_dist.low, x0_dist.high, log=True)
    search_space["x1"] = FloatDistribution(x1_dist.low, x1_dist.high, log=True)


class Problem(optunahub.benchmarks.BaseProblem):
    def __init__(self, problem_name: str) -> None:
        """Initialize the problem.
        Args:
            problem_name: Name of problem.

        Please refer to the reproblem repository for the details.
        https://github.com/ryojitanabe/reproblems
        """
        if problem_name not in unconstrained_problems:
            raise ValueError(
                f"problem_name must be in {list(unconstrained_problems.keys())}, "
                f"but got {problem_name}."
            )

        self.problem_name = problem_name

        self._problem = unconstrained_problems[problem_name]()

        search_space: dict[str, optuna.distributions.BaseDistribution] = {
            f"x{i}": FloatDistribution(low, high)
            for i, (low, high) in enumerate(zip(self._problem.lbound, self._problem.ubound))
        }
        # Hidden variable to reproduce the original work by setting it to False.
        self._enable_modification = True
        self._search_space = _modify_search_space_for_hard_problems(
            problem_name, search_space, enable=self._enable_modification
        )

        self._metric_names = list(_METRIC_NAMES[problem_name])
        n_objectives = self._problem.n_objectives
        assert len(self._metric_names) == n_objectives, (
            f"{problem_name} must have {n_objectives} objectives."
        )

        n_folded_constraints = self._problem.n_original_constraints
        assert len(_FOLDED_CONSTRAINT_NAMES[problem_name]) == n_folded_constraints, (
            f"{problem_name} must fold {n_folded_constraints} constraints into its objectives."
        )

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE] * self._problem.n_objectives

    @property
    def metric_names(self) -> list[str]:
        """Return the objective names in the order returned by ``evaluate``."""
        return self._metric_names.copy()

    @property
    def original_problem_name(self) -> str:
        """Return the name of the original problem, e.g. ``FourBarTruss`` for ``RE21``."""
        return _ORIGINAL_NAMES[self.problem_name]

    def evaluate(self, params: dict[str, float]) -> list[float]:
        x = np.array([params[name] for name in self._search_space])
        return self._problem.evaluate(x).tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)


class ConstrainedProblem(optunahub.benchmarks.BaseProblem):
    def __init__(self, problem_name: str, clip_constraints: bool = True) -> None:
        """Initialize the problem.
        Args:
            problem_name: Name of problem.
            clip_constraints:
                Whether to clip constraints by max(0, constraint_violation). Defaults to True
                following the original implementation.

        Please refer to the reproblem repository for the details.
        https://github.com/ryojitanabe/reproblems
        """
        if problem_name not in constrained_problems:
            raise ValueError(
                f"problem_name must be in {list(constrained_problems.keys())}, "
                f"but got {problem_name}."
            )

        self.problem_name = problem_name
        self.clip_constraints = clip_constraints

        self._problem = constrained_problems[problem_name]()

        search_space = {
            f"x{i}": FloatDistribution(low, high)
            for i, (low, high) in enumerate(zip(self._problem.lbound, self._problem.ubound))
        }
        # Hidden variable to reproduce the original work by setting it to False.
        self._enable_modification = True
        self._search_space = _modify_search_space_for_hard_problems(
            problem_name, search_space, enable=self._enable_modification
        )
        self._metric_names = list(_METRIC_NAMES[problem_name])
        n_objectives = self._problem.n_objectives
        assert len(self._metric_names) == n_objectives, (
            f"{problem_name} must have {n_objectives} objectives."
        )

        self._constraint_names = list(_CONSTRAINT_NAMES[problem_name])
        n_constraints = self._problem.n_constraints
        assert len(self._constraint_names) == n_constraints, (
            f"{problem_name} must have {n_constraints} constraints."
        )

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE] * self._problem.n_objectives

    @property
    def metric_names(self) -> list[str]:
        """Return the objective names in the order returned by ``evaluate``."""
        return self._metric_names.copy()

    @property
    def constraint_names(self) -> list[str]:
        """Return the constraint names used as the keys of ``evaluate_constraints``."""
        return self._constraint_names.copy()

    @property
    def original_problem_name(self) -> str:
        """Return the name of the original problem, e.g. ``TwoBarTruss`` for ``CRE21``."""
        return _ORIGINAL_NAMES[self.problem_name]

    def evaluate(self, params: dict[str, float]) -> list[float]:
        x = np.array([params[name] for name in self._search_space])
        f, _ = self._problem.evaluate(x)
        return f.tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def evaluate_constraints(self, params: dict[str, float]) -> dict[str, float]:
        x = np.array([params[name] for name in self._search_space])
        _, g = self._problem.evaluate(x)
        if self.clip_constraints:
            g = np.maximum(0.0, g)
        return dict(zip(self._constraint_names, g.tolist()))


if TYPE_CHECKING:
    for constrained_problem_cls in constrained_problems.values():
        _0: BaseConstrainedREBenchmark = constrained_problem_cls()
    for unconstrained_problem_cls in unconstrained_problems.values():
        _1: BaseUnconstrainedREBenchmark = unconstrained_problem_cls()
