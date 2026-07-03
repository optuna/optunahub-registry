from __future__ import annotations

import optuna
import optunahub


module = optunahub.load_module(
    package="visualization/plot_brute_force_tree",
)


def objective(trial: optuna.Trial) -> float:
    c = trial.suggest_categorical("c", ["float", "int"])
    if c == "float":
        return trial.suggest_float("x", 1, 3, step=0.5)
    else:
        a = trial.suggest_int("a", 1, 3)
        b = trial.suggest_int("b", a, 3)
        return a + b


study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(seed=42))
study.optimize(objective, n_trials=30)

fig = module.plot_brute_force_tree(study)
fig.write_html("brute_force_tree.html")
