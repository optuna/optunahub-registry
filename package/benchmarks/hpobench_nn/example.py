"""
This example is only for sampler.
You can verify your sampler code using this file as well.
Please feel free to remove this file if necessary.
"""

from __future__ import annotations

import optuna
import optunahub


# TODO: Change package_name to test your package.
package_name = "benchmarks/hpobench_nn"
test_local = True

if test_local:
    hpobench = optunahub.load_local_module(
        package=package_name,
        registry_root="./package",
    )
else:
    hpobench = optunahub.load_module(
        package=package_name, repo_owner="nabenabe0928", ref="add-hpobench"
    )

problem = hpobench.Problem(hpobench.Problem.available_dataset_names[0])
study = optuna.create_study()
study.optimize(problem, n_trials=30)
print(study.best_trials)
