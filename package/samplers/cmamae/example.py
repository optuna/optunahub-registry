import optuna
import optunahub

from sampler import CmaMaeSampler

# TODO: Replace above import with this.
#  module = optunahub.load_module("samplers/pyribs")
#  PyribsSampler = module.PyribsSampler


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return -(x**2 + y**2) + 2, x, y


if __name__ == "__main__":
    sampler = CmaMaeSampler(
        param_names=["x", "y"],
        archive_dims=[20, 20],
        archive_ranges=[(-10, 10), (-10, 10)],
        archive_learning_rate=0.1,
        archive_threshold_min=-10,
        n_emitters=15,
        emitter_x0={
            "x": 5,
            "y": 5
        },
        emitter_sigma0=0.1,
        emitter_batch_size=36,
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)
    print(study.best_trial.params)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("cmamae_optimization_history.png")
