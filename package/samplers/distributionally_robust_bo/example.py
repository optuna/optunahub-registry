from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return x**2 + y**2


if __name__ == "__main__":
    # Load the distributionally robust sampler from OptunaHub
    sampler = optunahub.load_module(
        package="samplers/distributionally_robust_bo"
    ).DistributionallyRobustSampler(epsilon_t=0.15, h=5.0, alpha=0.53)

    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=30)

    print("\nBest trials:")
    print(study.best_trials)
