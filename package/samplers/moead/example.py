import numpy as np
import optuna
import optunahub

from moead import MOEAdSampler


def objective(trial: optuna.Trial) -> tuple[float, float, float]:
    n_variables = 3

    x = np.array([trial.suggest_float(f"x{i}", 0, 1) for i in range(n_variables)])
    n = 10
    g = 100 * (n - 2) + 100 * np.sum((x - 0.5) ** 2 - np.cos(20 * np.pi * (x - 0.5)))

    f1 = (1 + g) * x[0] * x[1]
    f2 = (1 + g) * x[0] * (1 - x[1])
    f3 = (1 + g) * (1 - x[0])

    return f1, f2, f3


if __name__ == "__main__":
    # mod = optunahub.load_module("samplers/moea_d")
    # sampler = mod.MOEAdSampler()
    seed = 42
    population_size = 50
    n_trials = 1000
    crossover = optuna.samplers.nsgaii.BLXAlphaCrossover()
    samplers = [
        optuna.samplers.RandomSampler(seed=seed),
        # optuna.samplers.NSGAIIISampler(
        #     seed=seed, population_size=population_size, crossover=crossover
        # ),
        MOEAdSampler(
            seed=seed,
            population_size=population_size,
            n_neighbors=population_size // 5,
            scalar_aggregation_func="tchebycheff",
            crossover=crossover,
        ),
    ]
    studies = []
    for sampler in samplers:
        study = optuna.create_study(
            sampler=sampler,
            study_name=f"{sampler.__class__.__name__}",
            directions=["minimize", "minimize", "minimize"],
        )
        study.optimize(objective, n_trials=n_trials)
        studies.append(study)

    # optuna.visualization.plot_pareto_front(studies[0]).show()

    m = optunahub.load_module("visualization/plot_pareto_front_multi")
    fig = m.plot_pareto_front(studies)
    fig.show()
