import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0, 1)

    return x


if __name__ == "__main__":
    mod = optunahub.load_module("samplers/simulated_annealing")

    sampler = mod.SimulatedAnnealingSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)

    print(study.best_trial.value, study.best_trial.params)
