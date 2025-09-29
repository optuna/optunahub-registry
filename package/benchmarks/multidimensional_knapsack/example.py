from __future__ import annotations

import optuna
import optunahub


mkp_module = optunahub.load_module("benchmarks/multidimensional_knapsack")

problem = mkp_module.Problem(n_items=15, n_dimensions=3, seed=42)
study = optuna.create_study(directions=problem.directions)
study.optimize(problem, n_trials=100)
print(study.best_trial)