"""Minimal working example for the HierarchicalTPESampler.

The objective has a conditional search space: the categorical ``x`` gates whether ``y`` or
``z`` is requested, and the always-present ``t`` is coupled to whichever is active.
"""

from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    """Conditional objective: ``x`` gates ``y`` vs ``z``, each compared to the always-present ``t``.

    Args:
        trial: The Optuna trial to evaluate.

    Returns:
        The objective value to minimize.
    """
    x = trial.suggest_categorical("x", ["A", "B"])
    t = trial.suggest_float("t", -2, 2)
    if x == "A":
        y = trial.suggest_float("y", 1, 2)
        return (y - t) ** 2
    z = trial.suggest_float("z", -2, 1)
    return (z - t) ** 2


if __name__ == "__main__":
    HierarchicalTPESampler = optunahub.load_local_module(
        package="samplers/hierarchical_tpe", registry_root="./package"
    ).HierarchicalTPESampler

    # By default the conditional structure is learned automatically:
    sampler = HierarchicalTPESampler(seed=0)

    # Optionally, state the branching exactly via conditional_fn (mirrors the objective). It
    # receives the parameters chosen so far (here e.g. {"x": "A", "t": 0.3}, external values as
    # in trial.params) and returns the names of the parameters the objective requests next:
    def conditional_fn(params: dict) -> list[str]:
        return ["y"] if params["x"] == "A" else ["z"]

    sampler = HierarchicalTPESampler(seed=0, conditional_fn=conditional_fn)

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=50)
    print(study.best_params, study.best_value)
