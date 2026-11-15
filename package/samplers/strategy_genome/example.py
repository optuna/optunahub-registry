from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    kind = trial.suggest_categorical("kind", ["a", "b"])
    return (x - 3) ** 2 + (y + 2) ** 2 + (0 if kind == "b" else 5)


if __name__ == "__main__":
    module = optunahub.load_module(package="samplers/strategy_genome")
    sampler = module.GeneticStrategySampler(population_size=30, seed=1)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=300)
    print(study.best_params, study.best_value)
