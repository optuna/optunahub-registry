import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_float("y", -100, 100)
    return x**2 + y**2


def main() -> None:
    mod = optunahub.load_module("samplers/implicit_natural_gradient")

    sampler = mod.ImplicitNaturalGradientSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=200)

    print(study.best_trial.value, study.best_trial.params)


if __name__ == "__main__":
    main()
