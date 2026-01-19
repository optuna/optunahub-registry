import optuna
import optunahub


n_objs = 4
seed = 42
population_size = 50
n_trials = 1000

wfg = optunahub.load_module("benchmarks/wfg")
wfg1 = wfg.Problem(function_id=1, n_objectives=n_objs, dimension=10)

mod = optunahub.load_module("samplers/hype")
mutation = mod.PolynomialMutation()
crossover = optuna.samplers.nsgaii.SBXCrossover()

samplers = [
    mod.HypESampler(
        population_size=population_size,
        n_samples=4096,
        hypervolume_method="auto",
        mutation=mutation,
        crossover=crossover,
        seed=seed,
    ),
    optuna.samplers.NSGAIIISampler(
        population_size=population_size, crossover=crossover, seed=seed
    ),
]
studies = []
for sampler in samplers:
    study = optuna.create_study(
        sampler=sampler,
        study_name=f"{sampler.__class__.__name__}",
        directions=["minimize"] * n_objs,
    )
    study.optimize(wfg1, n_trials=n_trials)
    studies.append(study)

reference_point = [3 * (i + 1) for i in range(n_objs)]
m = optunahub.load_module("visualization/plot_hypervolume_history_multi")
fig = m.plot_hypervolume_history(studies, reference_point)
fig.show()
