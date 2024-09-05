import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    # ZDT1
    n_variables = 30

    x = np.array([trial.suggest_float(f"x{i}", 0, 1) for i in range(n_variables)])
    g = 1 + 9 * np.sum(x[1:]) / (n_variables - 1)
    f1 = x[0]
    f2 = g * (1 - (f1 / g) ** 0.5)

    return f1, f2


if __name__ == "__main__":
    mod = optunahub.load_module("samplers/moead")

    seed = 42
    population_size = 100
    n_trials = 10000
    crossover = optuna.samplers.nsgaii.BLXAlphaCrossover()
    samplers = [
        optuna.samplers.RandomSampler(seed=seed),
        optuna.samplers.NSGAIISampler(
            seed=seed,
            population_size=population_size,
            crossover=crossover,
        ),
        mod.MOEADSampler(
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
            directions=["minimize", "minimize"],
        )
        study.optimize(objective, n_trials=n_trials)
        studies.append(study)

        optuna.visualization.plot_pareto_front(study).show()

    m = optunahub.load_module("visualization/plot_pareto_front_multi")
    fig = m.plot_pareto_front(studies)
    fig.show()
