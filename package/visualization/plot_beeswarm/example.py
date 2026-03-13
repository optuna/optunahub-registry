"""Example usage of the beeswarm plot for OptunaHub."""

import matplotlib
import matplotlib.pyplot as plt
import optuna
import optunahub


matplotlib.use("Agg")

# Suppress Optuna logs for cleaner output.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load the beeswarm plot module from this branch.
mod = optunahub.load_module(
    package="visualization/plot_beeswarm",
)
plot_beeswarm = mod.plot_beeswarm


def objective(trial: optuna.trial.Trial) -> float:
    """Multi-parameter objective with clear monotonic relationships."""
    x = trial.suggest_float("x", 0.0, 10.0)
    y = trial.suggest_float("y", 0.0, 10.0)
    z = trial.suggest_float("z", 0.0, 10.0)
    w = trial.suggest_float("w", 0.0, 10.0)
    # x has the strongest influence, w the weakest.
    return 1.0 * x + 0.5 * y + 0.1 * z + 0.01 * w


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=500)

    fig, ax, cbar = plot_beeswarm(study)
    plt.savefig("beeswarm.png", dpi=150, bbox_inches="tight")
    print("Saved beeswarm.png")
