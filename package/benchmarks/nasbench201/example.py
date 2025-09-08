from __future__ import annotations

import optuna
import optunahub


hpobench = optunahub.load_module("benchmarks/nasbench201")
problem = hpobench.Problem(dataset_id=0)
study = optuna.create_study(directions=problem.directions)
study.optimize(problem, n_trials=30)
print(study.best_trial)
