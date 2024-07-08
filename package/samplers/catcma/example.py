import numpy as np
import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
import optunahub


def objective(trial: optuna.Trial) -> float:
    x1 = trial.suggest_float("x1", -1, 1)
    x2 = trial.suggest_float("x2", -1, 1)
    x3 = trial.suggest_float("x3", -1, 1)
    X = np.array([x1, x2, x3])

    c1 = trial.suggest_categorical("c1", [0, 1, 2])
    c2 = trial.suggest_categorical("c2", [0, 1, 2])
    c3 = trial.suggest_categorical("c3", [0, 1, 2])
    C = np.array([c1, c2, c3])

    return sum(X**2) + len(C) - sum(C == 0)


if __name__ == "__main__":
    mod = optunahub.load_module(
        package="samplers/catcma",
    )
    CatCmaSampler = mod.CatCmaSampler

    study = optuna.create_study(
        sampler=CatCmaSampler(
            search_space={
                "x1": FloatDistribution(-1, 1),
                "x2": FloatDistribution(-1, 1),
                "x3": FloatDistribution(-1, 1),
                "c1": CategoricalDistribution([0, 1, 2]),
                "c2": CategoricalDistribution([0, 1, 2]),
                "c3": CategoricalDistribution([0, 1, 2]),
            }
        )
    )
    study.optimize(objective, n_trials=20)
    print(study.best_params)

    # You can omit the search space definition before optimization.
    # Then, the search space will be estimated during the first trial.
    # In this case, independent_sampler (default: RandomSampler) will be used instead of the CatCma algorithm for the first trial.
    study = optuna.create_study(sampler=CatCmaSampler())
    study.optimize(objective, n_trials=20)
    print(study.best_params)
