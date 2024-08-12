import optuna
import optunahub


if __name__ == "__main__":
    samplers = [
        optuna.samplers.RandomSampler(),
        optuna.samplers.TPESampler(),
        optuna.samplers.CmaEsSampler(),
    ]

    mod = optunahub.load_module("samplers/implicit_natural_gradient")
    EnsembledSampler = mod.EnsembledSampler
    study = optuna.create_study(sampler=EnsembledSampler(samplers))

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0, 1)
        y = trial.suggest_float("y", 0, 1)
        return x + y

    study.optimize(objective, n_trials=20)
    print(study.best_params)
