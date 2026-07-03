from __future__ import annotations

import optuna
import optunahub


module = optunahub.load_module("visualization/plot_brute_force_tree")


def objective(trial: optuna.Trial) -> float:
    c = trial.suggest_categorical("c", ["float", "int"])
    if c == "float":
        return trial.suggest_float("x", 1, 3, step=0.5)
    else:
        a = trial.suggest_int("a", 1, 3)
        b = trial.suggest_int("b", a, 3)
        if b == a + 1:
            raise optuna.TrialPruned
        return a + b


study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(seed=42))
study.optimize(objective, n_trials=4)
c_dist = optuna.distributions.CategoricalDistribution(["float", "int"])
a_dist = optuna.distributions.IntDistribution(low=1, high=3)
study.enqueue_trial({"c": "int"})
study.ask(fixed_distributions={"c": c_dist, "a": a_dist})

fig = module.plot_brute_force_tree(study)
fig.show()
