"""
This example is only for sampler.
You can verify your sampler code using this file as well.
Please feel free to remove this file if necessary.
"""

from __future__ import annotations

import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    a = trial.suggest_categorical("a", ["c1", "c2"])
    if a == "c1":
        return np.exp(-((x - 3) ** 2))
    else:
        return np.exp(-((x + 3) ** 2))


# TODO: Change package_name to test your package.
package_name = "package/samplers/categorical_thompson_sampler"
test_local = True

if test_local:
    # This is an example of how to load a sampler from your local optunahub-registry.
    sampler = optunahub.load_local_module(
        package=package_name,
        registry_root="/Users/sammcd/locgit/optunahub-registry",  # Path to the root of the optunahub-registry.
    ).CategoricalThompsonSampler()
else:
    # This is an example of how to load a sampler from your fork of the optunahub-registry.
    # Please remove repo_owner and ref arguments before submitting a pull request.
    sampler = optunahub.load_module(
        package=package_name, repo_owner="Your GitHub Account ID", ref="Your Git Branch Name"
    ).YourSampler()

study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=30)
print(study.best_trials)
