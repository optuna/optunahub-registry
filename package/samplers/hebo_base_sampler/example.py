import time

import optuna
import optunahub


module = optunahub.load_module("samplers/hebo_base_sampler")
HEBOSampler = module.HEBOSampler


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    time.sleep(1.0)
    return x**2 + y**2


if __name__ == "__main__":
    sampler = HEBOSampler(constant_liar=True)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100, n_jobs=2)
    print(study.best_trial.params)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("hebo_optimization_history.png")
