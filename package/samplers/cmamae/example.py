import optuna
from optuna.study import StudyDirection
import optunahub


module = optunahub.load_module("samplers/cmamae")
CmaMaeSampler = module.CmaMaeSampler


def objective(trial: optuna.trial.Trial) -> tuple[float, float, float]:
    """Returns an objective followed by two measures."""
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return x**2 + y**2, x, y


if __name__ == "__main__":
    sampler = CmaMaeSampler(
        param_names=["x", "y"],
        archive_dims=[20, 20],
        archive_ranges=[(-1, 1), (-1, 1)],
        archive_learning_rate=0.1,
        archive_threshold_min=-10,
        n_emitters=1,
        emitter_x0={
            "x": 0,
            "y": 0,
        },
        emitter_sigma0=0.1,
        emitter_batch_size=20,
    )
    study = optuna.create_study(
        sampler=sampler,
        directions=[
            StudyDirection.MINIMIZE,
            # The remaining directions are for the measures, which do not have
            # an optimization direction. However, we set MINIMIZE as a
            # placeholder direction.
            StudyDirection.MINIMIZE,
            StudyDirection.MINIMIZE,
        ],
    )
    study.optimize(objective, n_trials=10000)
