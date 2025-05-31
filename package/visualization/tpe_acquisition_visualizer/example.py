from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optunahub


module = optunahub.load_module(
    package="visualization/tpe_acquisition_visualizer",
)


def objective(trial: optuna.trial.Trial) -> float:
    x_range = (-10, 50)
    x = trial.suggest_uniform("x", x_range[0], x_range[1])
    return np.abs(x) - 10 * np.cos(1 * x) + 10


seed = 42
tpe_acquisition_visualizer = module.TPEAcquisitionVisualizer()

n_startup_trials = 10
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(
        consider_prior=True,
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=False,
        n_startup_trials=n_startup_trials,
        n_ei_candidates=24,
        seed=seed,
        multivariate=False,
        group=False,
        constant_liar=False,
    ),
    direction="minimize",
)

study.optimize(objective, n_trials=100, callbacks=[tpe_acquisition_visualizer])

param_name = "x"
for trial in study.trials:
    if trial.number < n_startup_trials:
        continue
    fig = tpe_acquisition_visualizer.plot(study, trial.number, param_name)
    fig.savefig(f"{param_name}_{trial.number}.png")
    plt.close(fig)
