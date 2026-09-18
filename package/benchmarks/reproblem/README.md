---
author: Shuhei Watanabe
title: Real-World Multi-Objective Optimization Benchmark Problems (RE Problem Suite)
description: A collection of real-world (single- and multi-objective, constrained and unconstrained) optimization benchmark problems from the RE problem suite. This package is a wrapper of a re-implementation of the reproblems repository.
tags: [benchmark, multi-objective, real-world, constrained optimization, RE]
optuna_versions: [5.0.0]
license: MIT License
---

## Abstract

This package provides the real-world multi-objective optimization benchmark problems (the RE problem suite) introduced in [An Easy-to-use Real-world Multi-objective Problem Suite](https://arxiv.org/abs/2009.12867).
The original benchmark implementation is available [here](https://github.com/ryojitanabe/reproblems).
This package serves as a wrapper for a re-implementation of the original benchmark, ported to Python from the reference C source code (`reproblem.c`).

Note that `ConstrainedProblem` relies on `optuna.trial.Trial.set_constraint`, which requires Optuna v5.0.0 or newer.

### Disclaimer

This benchmark collection modified some parts of the original implementation:

- The constraint clipping can be disabled. The original implementation hard-codes the clipping.
- The `CRE1X` series is added by exposing the constraint sum from the `RE2X` series, so `CRE1X` does not belong to the contribution of the original paper.
- The search space of `CRE21` and `RE31` use the log scale for `x0` and `x1` because the feasiblity ratio is extremely low without this modification. You can also disable it by setting `<instance name>._enable_modification = False`.

## APIs

- `Problem(problem_name: str)`

  - `problem_name`: The name of an unconstrained benchmark problem. Available names are `RE21`, `RE22`, `RE23`, `RE24`, `RE25`, `RE31`, `RE32`, `RE33`, `RE34`, `RE35`, `RE36`, `RE37`, `RE41`, `RE42`, `RE61`, and `RE91`. Note that some of these problems internally reformulate their original constraints into an additional penalty objective, so they remain unconstrained from Optuna's perspective.
  - Attributes:
    - `search_space`: Return the search space.
      - Returns: `dict[str, optuna.distributions.BaseDistribution]`
    - `reference_point`: Return the reference point.
      - Returns: `list[float]`
    - `directions`: Return the optimization directions.
      - Returns: `list[optuna.study.StudyDirection]`
    - `metric_names`: Return the objective names in the order returned by `evaluate`.
      - Returns: `list[str]` of length `self.n_objectives`.
    - `original_problem_name`: Return the name of the original problem this problem is derived from, as in Table 1 of the paper, e.g. `FourBarTruss` for `RE21`.
      - Returns: `str`
    - `evaluate(params: dict[str, float])`: Evaluate the objective function given a dictionary of parameters.
      - Args:
        - `params`: A dictionary representing decision variables like `{"x0": x0_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `self.n_variables`.
      - Returns: List of length `self.n_objectives`.

- `ConstrainedProblem(problem_name: str, clip_constraints: bool = True)`

  - `problem_name`: The name of a constrained benchmark problem. Available names are `CRE12`, `CRE13`, `CRE14`, `CRE15`, `CRE21`, `CRE22`, `CRE23`, `CRE24`, `CRE25`, `CRE31`, `CRE32`, and `CRE51`.
  - `clip_constraints`: Whether to clip constraints by `max(0, constraint_violation)`. Defaults to `True` following the original implementation.
  - Attributes:
    - `search_space`: Return the search space.
      - Returns: `dict[str, optuna.distributions.BaseDistribution]`
    - `reference_point`: Return the reference point.
      - Returns: `list[float]`
    - `directions`: Return the optimization directions.
      - Returns: `list[optuna.study.StudyDirection]`
    - `metric_names`: Return the objective names in the order returned by `evaluate`.
      - Returns: `list[str]` of length `self.n_objectives`.
    - `original_problem_name`: Return the name of the original problem this problem is derived from, as in Table 1 of the paper, e.g. `FourBarTruss` for `RE21`.
      - Returns: `str`
    - `constraint_names`: Return the constraint names used as the keys of `evaluate_constraints`. This property is only available in `ConstrainedProblem`.
      - Returns: `list[str]` of length `self.n_constraints`.
    - `evaluate(params: dict[str, float])`: Evaluate the objective function given a dictionary of parameters.
      - Args:
        - `params`: A dictionary representing decision variables, with the same format as in `Problem.evaluate`.
      - Returns: List of length `self.n_objectives`.
    - `evaluate_constraints(params: dict[str, float])`: Evaluate the constraint functions and return the constraint values keyed by their names. A trial is considered feasible when all the values are zero or less.
      - Args:
        - `params`: A dictionary representing the decision variables, with the same format and value range as in `evaluate`.
      - Returns: `dict[str, float]` of length `self.n_constraints`.

The properties and functions of classes in [`reproblem.reproblem_original`](./reproblem_original) are also available, such as `lbound` and `ubound`.

## Objective and Constraint Names

Each RE problem models a different engineering problem, so unlike a synthetic suite, every problem has its own objectives.
The names below carry the `f_i` and `g_j` indices used by the [supplementary file](https://github.com/ryojitanabe/reproblems/blob/master/doc/re-supplementary_file.pdf), which defines each problem, so `metric_names` and `constraint_names` can be read directly against it.
Which names a problem uses can also be inspected at runtime via `problem.metric_names` and `problem.constraint_names`.

As Table 1 of the paper shows, an unconstrained problem and its constrained counterpart are derived from the same original problem, e.g. `RE31` and `CRE21` are both the `TwoBarTruss` problem, and the pair shares both the original problem name and the original objectives listed below.
The two variants differ in how they treat the original constraints: `Problem` folds them into an extra aggregated objective, whereas `ConstrainedProblem` exposes them through `evaluate_constraints`.
`CRE12`-`CRE15` are not in the paper's Table 1: `RE22`-`RE25` already fold real constraints into their violation objective, so this package exposes those same constraints as a constrained counterpart, following the naming convention of the pairs the paper does define.

### Objectives (all minimized)

The original names below are the ones listed in Table 1 of the paper and returned by `original_problem_name`.
Since the paper maximizes the annual cargo transport capacity of the `ConceptualMarine` problem, the corresponding objective is negated and prefixed with `negative_`.

| Original name            | `Problem` | `ConstrainedProblem` | Objective names shared by the pair                                                                                                                              |
| ------------------------ | --------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FourBarTruss`           | `RE21`    | -                    | `f1_structural_volume`, `f2_joint_displacement`                                                                                                                 |
| `ReinforcedConcreteBeam` | `RE22`    | `CRE12`              | `f1_total_cost`                                                                                                                                                 |
| `PressureVessel`         | `RE23`    | `CRE13`              | `f1_total_cost`                                                                                                                                                 |
| `HatchCover`             | `RE24`    | `CRE14`              | `f1_weight`                                                                                                                                                     |
| `CoilCompressionSpring`  | `RE25`    | `CRE15`              | `f1_volume`                                                                                                                                                     |
| `TwoBarTruss`            | `RE31`    | `CRE21`              | `f1_structural_weight`, `f2_member_ac_stress`                                                                                                                   |
| `WeldedBeam`             | `RE32`    | `CRE22`              | `f1_cost`, `f2_end_deflection`                                                                                                                                  |
| `DiscBrake`              | `RE33`    | `CRE23`              | `f1_brake_mass`, `f2_minimum_stopping_time`                                                                                                                     |
| `VehicleCrashworthiness` | `RE34`    | -                    | `f1_weight`, `f2_acceleration_characteristics`, `f3_toe_board_intrusion`                                                                                        |
| `SpeedReducer`           | `RE35`    | `CRE24`              | `f1_volume`, `f2_gear_shaft_stress`                                                                                                                             |
| `GearTrain`              | `RE36`    | `CRE25`              | `f1_gear_ratio_error`, `f2_max_gear_size`                                                                                                                       |
| `RocketInjector`         | `RE37`    | -                    | `f1_max_injector_face_temperature`, `f2_inlet_distance`, `f3_max_post_tip_temperature`                                                                          |
| `CarSideImpact`          | `RE41`    | `CRE31`              | `f1_car_weight`, `f2_pubic_force`, `f3_v_pillar_average_velocity`                                                                                               |
| `ConceptualMarine`       | `RE42`    | `CRE32`              | `f1_transportation_cost`, `f2_light_ship_weight`, `f3_negative_annual_cargo_transport_capacity`                                                                 |
| `WaterResourcePlanning`  | `RE61`    | `CRE51`              | `f1_drainage_network_cost`, `f2_storage_facility_cost`, `f3_treatment_facility_cost`, `f4_expected_flood_damage_cost`, `f5_expected_economic_loss_due_to_flood` |
| `CarCab`                 | `RE91`    | -                    | `f1_car_weight`, `f2_g1_violation`, `f3_g2_violation`, ..., `f9_g8_violation`                                                                                   |

The second objective of the `TwoBarTruss` problem is the stress of the member AC, which is the quantity that `g2` bounds, rather than a joint displacement.
A displacement would not share the 1e5 kPa limit that `g3` imposes on the stress of the member BC, and reference [27] of the paper, which is where this problem comes from, likewise gives the two objectives as the volume of the truss and the stress of the member AC.

In `Problem`, the original constraints are reformulated into one extra objective that sums their violations, named `f{n}_total_constraint_violation` where `n` is `self.n_objectives`.
For example, `Problem("RE31").metric_names` is `["f1_structural_weight", "f2_member_ac_stress", "f3_total_constraint_violation"]`.
The four problems whose original formulation has no constraint, namely `RE21`, `RE34`, `RE37`, and `RE91`, expose only the objectives listed above.
`RE91` is a special case: instead of aggregating the folded constraints, it keeps each of them as its own objective, so its violation objectives are named individually after the constraint they come from.

### Constraints (feasible when zero or less)

Neither the paper nor the supplementary file names a constraint, since the supplementary file defines each one only by its formula.
The descriptive part of each name below is therefore read off the constraint formula together with the original reference that the paper cites for the corresponding problem.

| `ConstrainedProblem` | `constraint_names`                                                                                                                                                                                                                                                                                                                           |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CRE12`              | `g1_flexural_capacity`, `g2_depth_to_width_ratio`                                                                                                                                                                                                                                                                                            |
| `CRE13`              | `g1_shell_thickness`, `g2_head_thickness`, `g3_working_volume`                                                                                                                                                                                                                                                                               |
| `CRE14`              | `g1_bending_stress`, `g2_shear_stress`, `g3_deflection`, `g4_buckling_stress`                                                                                                                                                                                                                                                                |
| `CRE15`              | `g1_shear_stress`, `g2_free_length`, `g3_coil_to_wire_diameter_ratio`, `g4_deflection_under_preload`, `g5_combined_deflection_clearance`, `g6_working_deflection`                                                                                                                                                                            |
| `CRE21`              | `g1_structural_weight`, `g2_member_ac_stress`, `g3_member_bc_stress`                                                                                                                                                                                                                                                                         |
| `CRE22`              | `g1_weld_shear_stress`, `g2_beam_bending_stress`, `g3_geometric_side`, `g4_buckling_load`                                                                                                                                                                                                                                                    |
| `CRE23`              | `g1_minimum_radial_thickness`, `g2_maximum_actuating_pressure`, `g3_maximum_temperature`, `g4_minimum_braking_torque`                                                                                                                                                                                                                        |
| `CRE24`              | `g1_gear_tooth_bending_stress`, `g2_gear_tooth_contact_stress`, `g3_shaft1_transverse_deflection`, `g4_shaft2_transverse_deflection`, `g5_pinion_pitch_diameter`, `g6_maximum_face_width_to_module`, `g7_minimum_face_width_to_module`, `g8_shaft1_length_clearance`, `g9_shaft2_length_clearance`, `g10_shaft1_stress`, `g11_shaft2_stress` |
| `CRE25`              | `g1_gear_ratio_error`                                                                                                                                                                                                                                                                                                                        |
| `CRE31`              | `g1_abdomen_load`, `g2_upper_viscous_criterion`, `g3_middle_viscous_criterion`, `g4_lower_viscous_criterion`, `g5_upper_rib_deflection`, `g6_middle_rib_deflection`, `g7_lower_rib_deflection`, `g8_pubic_symphysis_force`, `g9_b_pillar_velocity`, `g10_front_door_velocity`                                                                |
| `CRE32`              | `g1_length_to_beam_ratio`, `g2_length_to_depth_ratio`, `g3_length_to_draught_ratio`, `g4_draught_to_deadweight`, `g5_draught_to_depth`, `g7_maximum_deadweight`, `g6_minimum_deadweight`, `g8_froude_number`, `g9_metacentric_height`                                                                                                        |
| `CRE51`              | `g1`, `g2`, `g3`, `g4`, `g5`, `g6`, `g7`                                                                                                                                                                                                                                                                                                     |

`CRE51` is the exception that keeps the bare `g_j` indices.
Its seven constraints are regression surrogates that bound quantities which the original reference leaves unnamed, so there is nothing to read off them.

Note that `evaluate_constraints` preserves the order of the original implementation, which is not sorted by the constraint index for `CRE32`, as the table above shows.
This is a discrepancy between the paper and its implementations rather than a deliberate ordering: the C, Matlab, and Python versions all return `500000 - DWT` before `DWT - 3000`, whereas the supplementary file defines them the other way around as `g6` and `g7`.
The two are the halves of the single two-sided bound `3000 <= DWT <= 500000`, so neither order is more correct than the other, and the names above follow the implementation because that is the order the values come back in.

Note also that the original implementation returns the *violation* of each constraint rather than the constraint function value itself.
The supplementary file writes every constraint as `g_j(x) >= 0`, and the implementation converts it into `max(-g_j(x), 0)`, which is zero exactly when the constraint is satisfied and positive otherwise.
The returned values are therefore always zero or greater, and a trial is feasible when every value is zero, which is consistent with Optuna's convention that a constraint is satisfied when its value is zero or less.

### Constraints folded into `f{n}_total_constraint_violation`

Every problem listed in the table above that has original constraints exposes them through its constrained counterpart, so what `Problem` folds into the violation objective can always be looked up there.
`RE21`, `RE34`, and `RE37` have no original constraints at all, so their `Problem` has nothing folded in, and `RE91` is the special case described above that keeps each folded constraint as its own objective instead.

These names are documentation only, since `constraint_names` exists on `ConstrainedProblem` alone.

## Example

```python
from __future__ import annotations

import optuna
import optunahub


reproblem = optunahub.load_module("benchmarks/reproblem")
problem = reproblem.ConstrainedProblem("CRE21")
study = optuna.create_study(directions=problem.directions)
study.optimize(problem, n_trials=10)

if len(problem.directions) == 1:
    print(study.best_trial)
else:
    print(study.best_trials)
```

## Reference

```bibtex
@article{tanabe2020easy,
  title={An easy-to-use real-world multi-objective optimization problem suite},
  author={Tanabe, Ryoji and Ishibuchi, Hisao},
  journal={Applied Soft Computing},
  volume={89},
  pages={106078},
  year={2020},
  publisher={Elsevier}
}
```
