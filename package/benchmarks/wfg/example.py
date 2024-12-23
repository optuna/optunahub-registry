import optuna
import optunahub


wfg = optunahub.load_local_module("benchmarks/wfg", registry_root="../../")
wfg4 = wfg.Problem(function_id=4, num_objectives=2, num_variables=3, k=1)

study_pareto = optuna.create_study(study_name="ParetoFront", directions=wfg4.directions)
for x in wfg4.get_optimal_solutions(1000):  # Generate 1000 optimal solutions
    study_pareto.enqueue_trial(params={f"x{i}": x.phenome[i] for i in range(3)})
study_pareto.optimize(wfg4, n_trials=1000)

study_tpe = optuna.create_study(
    study_name="TPESampler",
    sampler=optuna.samplers.TPESampler(seed=42),
    directions=wfg4.directions,
)
study_tpe.optimize(wfg4, n_trials=1000)

optunahub.load_module("visualization/plot_pareto_front_multi").plot_pareto_front(
    [study_pareto, study_tpe]
).show()
