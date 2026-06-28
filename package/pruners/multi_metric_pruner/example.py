"""Examples of MultiMetricPruner with MultiMetricPrunerTrial."""

import optuna
import optunahub


module = optunahub.load_local_module("pruners/multi_metric_pruner", registry_root="package/")
MultiMetricPruner = module.MultiMetricPruner
MultiMetricPrunerTrial = module.MultiMetricPrunerTrial


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Multi-metric mode — report all metrics jointly at each step.
# should_prune() with no argument uses Pareto ranking over all metrics.
# ──────────────────────────────────────────────────────────────────────────────


def objective_multi(trial: optuna.Trial) -> tuple[float, float]:
    trial = MultiMetricPrunerTrial(trial)
    x = trial.suggest_float("x", -5.0, 5.0)

    for step in range(10):
        metric1 = (x - step * 0.1) ** 2
        metric2 = (x + step * 0.1) ** 2
        trial.report({"loss": metric1, "acc": metric2}, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return x**2, (x - 2.0) ** 2


study_multi = optuna.create_study(
    directions=["minimize", "minimize"],
    pruner=MultiMetricPruner(
        optuna.pruners.MedianPruner(n_startup_trials=3),
        metric_directions={"loss": "minimize", "acc": "minimize"},
    ),
)
study_multi.optimize(objective_multi, n_trials=30)
print(f"[Multi-metric] Completed trials: {len(study_multi.trials)}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Per-metric mode — report each metric independently at the same step,
# and prune by each metric name separately.
# This is useful when metrics have different directions or different importance,
# and you want the base pruner to evaluate each one on its own scale.
# ──────────────────────────────────────────────────────────────────────────────


def objective_per_metric(trial: optuna.Trial) -> tuple[float, float]:
    trial = MultiMetricPrunerTrial(trial)
    x = trial.suggest_float("x", -5.0, 5.0)

    for step in range(10):
        loss = (x - step * 0.1) ** 2
        acc = 1.0 / (1.0 + (x + step * 0.1) ** 2)
        # Report each metric independently — they are merged at the same step.
        trial.report({"loss": loss}, step)
        trial.report({"acc": acc}, step)
        # Prune if either metric individually warrants pruning.
        if trial.should_prune(metric_name="loss"):
            raise optuna.TrialPruned()
        if trial.should_prune(metric_name="acc"):
            raise optuna.TrialPruned()

    return x**2, 1.0 / (1.0 + (x - 2.0) ** 2)


study_per_metric = optuna.create_study(
    directions=["minimize", "maximize"],
    pruner=MultiMetricPruner(
        optuna.pruners.MedianPruner(n_startup_trials=3),
        metric_directions={"loss": "minimize", "acc": "maximize"},
    ),
)
study_per_metric.optimize(objective_per_metric, n_trials=30)
print(f"[Per-metric] Completed trials: {len(study_per_metric.trials)}")
