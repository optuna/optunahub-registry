"""Example: Meta-Learn TPE with warm-starting from related tasks.

Scenario: Optimize a function after already having optimized two related
functions. The meta-learning sampler transfers knowledge from the source
studies to speed up convergence on the target task.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna
import optunahub


def make_shifted_objective(shift: float) -> Callable[[optuna.Trial], float]:
    """Create a shifted Branin-like objective."""

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -5.0, 10.0)
        y = trial.suggest_float("y", 0.0, 15.0)
        a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi, 6.0, 10.0, 1 / (8 * np.pi)
        return float(
            a * (y - b * (x + shift) ** 2 + c * (x + shift) - r) ** 2
            + s * (1 - t) * np.cos(x + shift)
            + s
        )

    return objective


if __name__ == "__main__":
    # 1. Run source studies on related tasks.
    source_study_1 = optuna.create_study()
    source_study_1.optimize(make_shifted_objective(shift=0.0), n_trials=30)

    source_study_2 = optuna.create_study()
    source_study_2.optimize(make_shifted_objective(shift=0.5), n_trials=30)

    # 2. Create the meta-learning sampler with source studies.
    sampler = optunahub.load_module(package="samplers/meta_learn_tpe").MetaLearnTPESampler(
        source_studies=[source_study_1, source_study_2],
        seed=42,
    )

    # 3. Optimize the target task (slightly shifted).
    target_study = optuna.create_study(sampler=sampler)
    target_study.optimize(make_shifted_objective(shift=0.25), n_trials=50)

    print(f"Best value: {target_study.best_value:.4f}")
    print(f"Best params: {target_study.best_params}")
