---
author: Optuna Team
title: Single and Multi-objective Optimization Benchmark Problems Focusing on Human-Powered Aircraft Design
description: The benchmark problem for human-powered aircraft design introduced in the paper `Single and Multi-Objective Optimization Benchmark Problems Focusing on Human-Powered Aircraft Design`
tags: [benchmark, HPA, multi-objective, human-powered aircraft]
optuna_versions: [5.0.0]
license: MIT License
---

## Abstract

The benchmark for human-powered aircraft (hpa) design is introduced in the paper [Single and Multi-Objective Optimization Benchmark Problems Focusing on Human-Powered Aircraft Design](https://arxiv.org/abs/2312.08953).
The original benchmark is available [here](https://github.com/Nobuo-Namura/hpa).
This package serves as a wrapper for the original benchmark.

## APIs

### class `ConstrainedProblem(problem_name: str, n_div: int = 4, level: int = 0 )`

- `problem_name`: The name of a benchmark problem. All problem names and their explanations are provided [here](https://github.com/Nobuo-Namura/hpa?tab=readme-ov-file#benchmark-problem-definition).
- `n_div`: The wing segmentation number and alters the problem's dimension. It must be an integer greater than 0. Concretely, the number of sections in [this figure](https://github.com/Nobuo-Namura/hpa/blob/main/img/variables.jpg). The default value used in the paper is 4.
- `level`: The difficulty level of the problem. It must be in `[0, 1, 2]`

Note that `Problem` also receives the same set of arguments.

#### Method and Properties

- `search_space`: Return the search space.
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions.
  - Returns: `list[optuna.study.StudyDirection]`
- `metric_names`: Return the objective names in the order returned by `evaluate`.
  - Returns: `list[str]` of length `self.nf`.
- `constraint_names`: Return the constraint names used as the keys of `evaluate_constraints`. This property is only available in `ConstrainedProblem`.
  - Returns: `list[str]` of length `self.ng`.
- `evaluate(params: dict[str, float])`: Evaluate the objective function given a dictionary of parameters.
  - Args:
    - `params`: A dictionary representing decision variable like `{"x0": x1_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `self.nx`. `xn_value` must be a `float` in `[0, 1]`.
  - Returns: List of length `self.nf`.
- `evaluate_constraints(params: dict[str, float])`: Evaluate the constraint functions and return the constraint function values keyed by their names. This method is only available in `ConstrainedProblem`.
  - Args:
    - `params`: A dictionary representing the decision variables, with the same format and value range as in evaluate.
  - Returns: Dictionary of length `self.ng`. A trial is feasible when every value is zero or less.

The properties and functions of classes in [`hpa.problem`](https://hub.optuna.org/benchmarks/hpa/hpa_original) are also available such as `nx`.

## Objective and Constraint Names

Each problem uses a subset of the 11 fundamental objectives and the 5 constraints defined in Table 1 of the paper.
The names below carry the paper's `f_i` and `g_j` indices as a prefix, so `metric_names` and `constraint_names` can be read directly against Table 2 of the paper.
Which subset a problem uses can also be inspected at runtime via `problem.metric_names` and `problem.constraint_names`.

### Objectives (all minimized)

Since the paper maximizes the cruise speed, the wing efficiency, and the payload, the corresponding objectives are negated and prefixed with `negative_`.

| Name                          | Paper                                       | Unit |
| ----------------------------- | ------------------------------------------- | ---- |
| `f1_required_power`           | $f_1 = P$                                   | W    |
| `f2_drag`                     | $f_2 = D$                                   | N    |
| `f3_negative_cruise_speed`    | $f_3 = -V$                                  | m/s  |
| `f4_max_wingtip_deflection`   | $f_4 = \max(\|\delta\|, \|\delta_{park}\|)$ | m    |
| `f5_max_twist_angle`          | $f_5 = \Phi$                                | deg  |
| `f6_negative_wing_efficiency` | $f_6 = -E$                                  | -    |
| `f7_empty_weight`             | $f_7 = W_0$                                 | kg   |
| `f8_wing_span`                | $f_8 = B$                                   | m    |
| `f9_root_angle_of_attack`     | $f_9 = \alpha_0$                            | deg  |
| `f10_wire_tension`            | $f_{10} = T$                                | N    |
| `f11_negative_payload`        | $f_{11} = -W_p$                             | kg   |

### Constraints (feasible when zero or less)

| Name                           | Paper                                           | Unit |
| ------------------------------ | ----------------------------------------------- | ---- |
| `g1_max_strain`                | $g_1 = n_m n_s \epsilon_{max} / \epsilon_u - 1$ | -    |
| `g2_wingtip_dihedral_angle`    | $g_2 = B (\sin\gamma - \sin\gamma_u) / 2$       | m    |
| `g3_parked_wingtip_deflection` | $g_3 = -\delta_{park}$                          | m    |
| `g4_required_power`            | $g_4 = P - P_{max}$                             | W    |
| `g5_min_cruise_speed`          | $g_5 = 1 - (V / V_{min})^3$                     | -    |

Note that `evaluate_constraints` preserves the order of the original implementation, which is not sorted by the constraint index.
For example, `ConstrainedProblem("HPA131").constraint_names` is `["g1_max_strain", "g3_parked_wingtip_deflection", "g2_wingtip_dihedral_angle"]`.

In the unconstrained problems, the constraints are folded into the objectives as penalty terms following Eq. (5) of the paper, so `Problem` exposes only `metric_names`.

## Installation

The dependencies can be installed via:

```shell
pip install pandas scipy optunahub
```

Or you can install the required packages from optunahub as well.

```shell
pip install -r https://hub.optuna.org/benchmarks/hpa/requirements.txt
```

## Example

```Python
from __future__ import annotations

import optuna
import optunahub


hpa = optunahub.load_module("benchmarks/hpa")
problem = hpa.ConstrainedProblem("HPA131") 
study = optuna.create_study(directions=problem.directions)
study.optimize(problem, n_trials=10)


if len(problem.directions) == 1:
    print(study.best_trial)
else:
    print(study.best_trials)
```

## Reference

```bibtex
@inproceedings{namura2025single,
  title={Single and multi-objective optimization benchmark problems focusing on human-powered aircraft design},
  author={Namura, Nobuo},
  booktitle={International Conference on Evolutionary Multi-Criterion Optimization},
  pages={195--210},
  year={2025},
  organization={Springer}
}
```
