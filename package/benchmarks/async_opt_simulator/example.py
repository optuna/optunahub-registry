from __future__ import annotations

import optuna
import optunahub


sim = optunahub.load_module("benchmarks/async_opt_simulator").AsyncOptBenchmarkSimulator(
    n_workers=4
)
Problem = optunahub.load_module("benchmarks/hpolib").Problem
problem = Problem(dataset_id=0, metric_names=["val_loss"])
runtime_func = Problem(dataset_id=0, metric_names=["train_time"])
study = optuna.create_study(directions=problem.directions)
sim.optimize(study=study, problem=problem, runtime_func=lambda t: runtime_func(t)[0], n_trials=100)
print(sim.get_results_from_study(study))
