"""Example usage of CMA-ES Animation for Optunahub
Note : Required FFmpeg
"""

import numpy as np
import optuna
import optunahub


# Suppress Optuna logs for cleaner output.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load the CMA-ES plot module from this branch.
mod = optunahub.load_module(
    package="visualization/plot_cmaes",
)


plot_cmaes = mod.plot_cmaes


def objective(trial: optuna.trial.Trial) -> float:
    """Quadratic function"""

    x1 = trial.suggest_float("x1", -4, 4)
    x2 = trial.suggest_float("x2", -4, 4)

    return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2


def quadratic_contour(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Contour function representing the objective landscape."""

    return np.log(((x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2) + 1)


if __name__ == "__main__":
    sampler = optuna.samplers.CmaEsSampler(seed=0)  # for reproducibility
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=250)

    # With Known Contour Function
    animation_with_contour_function = plot_cmaes(study, contour_function=quadratic_contour)
    animation_with_contour_function.save("Cmaes_contour_function.mp4", writer="ffmpeg")
    print("Saved Cmaes_contour_function.mp4")

    # without Contour Function
    animation_without_contour_function = plot_cmaes(study)
    animation_without_contour_function.save("Cmaes_without_contour_function.mp4", writer="ffmpeg")
    print("Saved Cmaes_without_contour_function.mp4")
