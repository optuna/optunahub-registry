import optuna
import optunahub


module = optunahub.load_module(package="samplers/tpe_cat_distance")


def hamming_distance(x: str, y: str) -> float:
    return float(x != y)


sampler = module.TPESampler(
    n_startup_trials=2,
    seed=7,
    categorical_distance_func={"optimizer": hamming_distance},
)


def objective(trial: optuna.Trial) -> float:
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    return float(optimizer != "adam")


study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective, n_trials=10)
print(study.best_params)
