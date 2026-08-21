"""Minimal working example for the max-value entropy search sampler."""

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


if __name__ == "__main__":
    mod = optunahub.load_module("samplers/gp_mes")

    # The Gumbel approximation of the max-value distribution: cheaper, but biased upward.
    study = optuna.create_study(sampler=mod.MESSampler(max_value_sampler="gumbel", seed=42))
    study.optimize(objective, n_trials=50)
    print(f"gumbel    best value: {study.best_value:.5f}, params: {study.best_params}")

    # Joint posterior sampling, the default.
    study = optuna.create_study(sampler=mod.MESSampler(max_value_sampler="posterior", seed=42))
    study.optimize(objective, n_trials=50)
    print(f"posterior best value: {study.best_value:.5f}, params: {study.best_params}")
