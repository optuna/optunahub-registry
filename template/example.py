"""
This example is only for sampler.
You can verify your sampler code using this file as well.
Please feel free to remove this file if necessary.
"""

from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("x", -5, 5)
    return x**2 + y**2


# TODO: Change the variables here to test your package.
package_name = "samplers/your_sampler"
repo_owner = "Your GitHub Account Name"
ref = "Your Git Branch Name"
test_local = True

if test_local:
    sampler = optunahub.load_local_module(
        package=package_name, registry_root="./package/"
    ).YourSampler()
else:
    sampler = optunahub.load_module(
        package=package_name, repo_owner=repo_owner, ref=ref
    ).YourSampler()

study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=30)
print(study.best_trials)
