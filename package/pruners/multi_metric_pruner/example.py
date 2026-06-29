"""Examples of MultiMetricPruner with MultiMetricPrunerTrial."""

import optuna
import optunahub


module = optunahub.load_module("pruners/multi_metric_pruner")
MultiMetricPruner = module.MultiMetricPruner
MultiMetricPrunerTrial = module.MultiMetricPrunerTrial


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Multi-metric mode — report all metrics jointly at each step.
# should_prune() with no argument uses Pareto ranking over all metrics.
# ──────────────────────────────────────────────────────────────────────────────


def objective_multi(trial: optuna.Trial) -> tuple[float, float]:
    mmt = MultiMetricPrunerTrial(trial)
    x = mmt.suggest_float("x", -5.0, 5.0)

    for step in range(10):
        metric1 = (x - step * 0.1) ** 2
        metric2 = (x + step * 0.1) ** 2
        mmt.report({"loss": metric1, "acc": metric2}, step)
        if mmt.should_prune():
            raise optuna.TrialPruned()

    return x**2, (x - 2.0) ** 2


study_multi = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=MultiMetricPruner(
        optuna.pruners.MedianPruner(n_startup_trials=3),
        metric_directions={"loss": "minimize", "acc": "minimize"},
        joint=True,
    ),
)
study_multi.optimize(objective_multi, n_trials=100)
print(f"[Multi-metric] Completed trials: {len(study_multi.trials)}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Per-metric mode — report each metric independently at the same step,
# and prune by each metric name separately.
# This is useful when metrics have different directions or different importance,
# and you want the base pruner to evaluate each one on its own scale.
# ──────────────────────────────────────────────────────────────────────────────


def objective_per_metric(trial: optuna.Trial) -> tuple[float, float]:
    mmt = MultiMetricPrunerTrial(trial)
    x = mmt.suggest_float("x", -5.0, 5.0)

    for step in range(10):
        loss = (x - step * 0.1) ** 2
        acc = 1.0 / (1.0 + (x + step * 0.1) ** 2)
        # Reporting multiple metrics is iterated per-metric automatically when joint=False.
        mmt.report({"loss": loss, "acc": acc}, step)
        # Prune if either metric individually warrants pruning.
        if mmt.should_prune():
            raise optuna.TrialPruned()

    return x**2, 1.0 / (1.0 + (x - 2.0) ** 2)


study_per_metric = optuna.create_study(
    directions=["minimize", "maximize"],
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=MultiMetricPruner(
        {
            "loss": optuna.pruners.MedianPruner(n_startup_trials=3),
            "acc": optuna.pruners.MedianPruner(n_startup_trials=5),
        },
        metric_directions={"loss": "minimize", "acc": "maximize"},
        joint=False,
    ),
)
study_per_metric.optimize(objective_per_metric, n_trials=100)
print(f"[Per-metric] Completed trials: {len(study_per_metric.trials)}")


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3: Mixed-frequency per-metric mode — metrics have different computational
# costs and are therefore reported at different step intervals.
# train_loss is cheap and reported every step; val_loss is expensive and only
# evaluated every 5 steps.  should_prune() is called right after each report
# with metric_name= so only that metric's history is checked.
# ──────────────────────────────────────────────────────────────────────────────


def objective_mixed_freq(trial: optuna.Trial) -> tuple[float, float]:
    mmt = MultiMetricPrunerTrial(trial)
    x = mmt.suggest_float("x", -5.0, 5.0)

    for step in range(10):
        # Cheap metric — computed at every step.
        train_loss = (x - step * 0.1) ** 2
        mmt.report({"train_loss": train_loss}, step)
        if mmt.should_prune(metric_name="train_loss"):
            raise optuna.TrialPruned()

        # Expensive metric — computed only every 5 steps.
        if step % 5 == 0:
            val_loss = (x + step * 0.05) ** 2
            mmt.report({"val_loss": val_loss}, step)
            if mmt.should_prune(metric_name="val_loss"):
                raise optuna.TrialPruned()

    return x**2, (x - 2.0) ** 2


study_mixed = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=MultiMetricPruner(
        {
            "train_loss": optuna.pruners.MedianPruner(n_startup_trials=3),
            "val_loss": optuna.pruners.PercentilePruner(percentile=50.0, n_startup_trials=1),
        },
        metric_directions={"train_loss": "minimize", "val_loss": "minimize"},
        joint=False,
    ),
)
study_mixed.optimize(objective_mixed_freq, n_trials=100)
print(f"[Mixed-frequency] Completed trials: {len(study_mixed.trials)}")
