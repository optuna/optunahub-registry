import optuna
import optunahub


# Set objective function:
def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)

    return x**2 + y**2


if __name__ == "__main__":
    # Set up sampler:
    module = optunahub.load_module("samplers/confopt_sampler")  # type: ignore[attr-defined]
    sampler = module.ConfOptSampler(
        # Search space below must match the one defined in the objective function:
        search_space={
            "x": optuna.distributions.FloatDistribution(-10, 10),
            "y": optuna.distributions.IntDistribution(-10, 10),
        },
        # Number of random searches before switching to inferential search:
        n_startup_trials=10,
    )

    # Run study:
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)

    print(f"Best trial parameters: {study.best_trial.params}")
    print(f"Best trial value: {study.best_trial.value}")
