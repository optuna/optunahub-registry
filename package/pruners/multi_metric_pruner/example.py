"""Examples of MultiMetricPruner with MultiMetricPrunerTrial."""

import optuna
import optunahub


module = optunahub.load_local_module("pruners/multi_metric_pruner", registry_root="package/")
MultiMetricPruner = module.MultiMetricPruner
MultiMetricPrunerTrial = module.MultiMetricPrunerTrial


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Multi-metric mode — report both metrics jointly at each step.
# Pruning uses Pareto ranking over all intermediate value pairs.
# ──────────────────────────────────────────────────────────────────────────────


def objective_multi(trial: optuna.Trial) -> tuple[float, float]:
    trial = MultiMetricPrunerTrial(trial)
    x = trial.suggest_float("x", -5.0, 5.0)

    for step in range(10):
        metric1 = (x - step * 0.1) ** 2
        metric2 = (x + step * 0.1) ** 2
        trial.report([metric1, metric2], step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return x**2, (x - 2.0) ** 2


study_multi = optuna.create_study(
    directions=["minimize", "minimize"],
    pruner=MultiMetricPruner(
        optuna.pruners.MedianPruner(n_startup_trials=3),
        directions=["minimize", "minimize"],
    ),
)
study_multi.optimize(objective_multi, n_trials=30)
print(f"[Multi-metric] Completed trials: {len(study_multi.trials)}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Single-metric mode — report each metric independently by name.
# Pruning considers only the metric named in should_prune.
# ──────────────────────────────────────────────────────────────────────────────


def objective_single(trial: optuna.Trial) -> tuple[float, float]:
    trial = MultiMetricPrunerTrial(trial)
    x = trial.suggest_float("x", -5.0, 5.0)

    for step in range(10):
        loss = (x - step * 0.1) ** 2
        trial.report(loss, step, metric_name="loss")
        if trial.should_prune(metric_name="loss"):
            raise optuna.TrialPruned()

    return x**2, (x - 2.0) ** 2


study_single = optuna.create_study(
    directions=["minimize", "minimize"],
    pruner=MultiMetricPruner(
        optuna.pruners.MedianPruner(n_startup_trials=3),
        metric_directions={"loss": "minimize"},
    ),
)
study_single.optimize(objective_single, n_trials=30)
print(f"[Single-metric] Completed trials: {len(study_single.trials)}")
