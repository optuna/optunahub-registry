from __future__ import annotations

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
import optuna.study.study
import optunahub


def objective(x: float, y: float) -> float:
    return x**2 + y**2


def optuna_objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return objective(x, y)


if __name__ == "__main__":
    # You can specify the search space before optimization.
    # This allows the sampler to generate the initial simplex based on the specified search space at the first trial.
    search_space: dict[str, BaseDistribution] = {
        "x": FloatDistribution(-5, 5),
        "y": FloatDistribution(-5, 5),
    }
    module = optunahub.load_module(
        package="samplers/nelder_mead",
    )

    sampler = module.NelderMeadSampler(search_space, seed=123)
    study = optuna.create_study(sampler=sampler)
    # Ask-and-Tell style optimizaiton.
    for i in range(100):
        trial = study.ask(search_space)
        value = objective(**trial.params)
        study.tell(trial, value)
        print(
            f"Trial {trial.number} finished with values: {value} and parameters: {trial.params}. "
            f"Best it trial {study.best_trial.number} with value: {study.best_value}"
        )
    print(study.best_params, study.best_value)

    # study.optimize can be used with an Optuna-style objective function.
    sampler = module.NelderMeadSampler(search_space, seed=123)
    study = optuna.create_study(sampler=sampler)
    study.optimize(optuna_objective, n_trials=100)
    print(study.best_params, study.best_value)

    # Without the search_space argument, the search space is estimated during the first trial.
    # In this case, independent_sampler (default: RandomSampler) will be used instead of the Nelder-Mead algorithm for the first trial.
    sampler = module.NelderMeadSampler(seed=123)
    study = optuna.create_study(sampler=sampler)
    study.optimize(optuna_objective, n_trials=100)
    print(study.best_params, study.best_value)
