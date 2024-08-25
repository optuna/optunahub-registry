import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


if __name__ == "__main__":
    mod = optunahub.load_module("samplers/moea_d")

    sampler = mod.MOEAdSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)

    print(study.best_trials)
