from __future__ import annotations

import optuna
import optunahub


mkp_module = optunahub.load_module(
    "benchmarks/multidimensional_knapsack",
)

problem = mkp_module.Problem(n_items=100, n_dimensions=10, seed=42)
sampler = optuna.samplers.NSGAIISampler(constraints_func=problem.evaluate_constraints)
study = optuna.create_study(directions=problem.directions, sampler=sampler)
study.optimize(problem, n_trials=100)
print(study.best_trial)
