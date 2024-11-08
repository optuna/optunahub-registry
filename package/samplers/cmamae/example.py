import optuna
from optuna.study import StudyDirection
from sampler import CmaMaeSampler


# TODO: Replace above import with this.
#  module = optunahub.load_module("samplers/pyribs")
#  PyribsSampler = module.PyribsSampler


def objective(trial: optuna.trial.Trial) -> tuple[float, float, float]:
    """Returns an objective followed by two measures."""
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
        n_emitters=1,
        emitter_x0={"x": 5, "y": 5},
        emitter_sigma0=0.1,
        emitter_batch_size=5,
    )
    study = optuna.create_study(
        sampler=sampler,
        directions=[
            # pyribs maximizes objectives.
            StudyDirection.MAXIMIZE,
            # The remaining values are measures, which do not have an
            # optimization direction.
            # TODO: Currently, using StudyDirection.NOT_SET is not allowed as
            # Optuna assumes we either minimize or maximize.
            StudyDirection.MINIMIZE,
            StudyDirection.MINIMIZE,
        ],
    )
    study.optimize(objective, n_trials=100)

    # TODO: Visualization.
    #  fig = optuna.visualization.plot_optimization_history(study)
    #  fig.write_image("cmamae_optimization_history.png")
