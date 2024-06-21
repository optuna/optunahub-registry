import optuna
import optunahub


# module = optunahub.load_module("samplers/hebo")
module = optunahub.load_local_module("samplers/hebo", registry_root="./package")
HEBOSampler = module.HEBOSampler


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    return x**2 + y**2


if __name__ == "__main__":
    sampler = HEBOSampler(
        {
            "x": optuna.distributions.FloatDistribution(-10, 10),
            "y": optuna.distributions.IntDistribution(-10, 10),
        }
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)
    print(study.best_trial.params)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("hebo_optimization_history.png")
