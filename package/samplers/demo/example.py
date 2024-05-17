import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0, 1)

    return x


if __name__ == "__main__":
    module = optunahub.load_module("samplers/demo")
    sampler = module.DemoSampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=5)

    print(study.best_trial)
