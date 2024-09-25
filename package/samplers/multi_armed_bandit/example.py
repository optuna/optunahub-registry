import optuna
import optunahub


if __name__ == "__main__":
    module = optunahub.load_module(
        package="samplers/multi_armed_bandit",
    )
    sampler = module.MultiArmedBanditSampler()

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_categorical("arm_1", [1, 2, 3])
        y = trial.suggest_categorical("arm_2", [1, 2])

        return x + y

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)

    print(study.best_trial.value, study.best_trial.params)
