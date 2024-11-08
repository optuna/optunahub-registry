import optuna
import optunahub


module = optunahub.load_module("samplers/smac_sampler")
SMACSampler = module.SMACSampler


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    return x**2 + y**2


if __name__ == "__main__":
    n_trials = 100
    sampler = SMACSampler(
        {
            "x": optuna.distributions.FloatDistribution(-10, 10),
            "y": optuna.distributions.IntDistribution(-10, 10),
        },
        n_trials=n_trials,
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    print(study.best_trial.params)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("smac_optimization_history.png")
