import optuna
import optunahub


if __name__ == "__main__":

    def objective(trial: optuna.trial.Trial) -> None:
        x = trial.suggest_int("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return -(x**2 + y**2)

    module = optunahub.load_module(package="samplers/hill-climb-search")
    sampler = module.HillClimbingSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)

    print(study.best_trial.value, study.best_trial.params)
