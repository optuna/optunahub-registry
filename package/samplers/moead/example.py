import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


if __name__ == "__main__":
    mod = optunahub.load_module("samplers/moead")

    population_size = 100
    n_trials = 1000
    crossover = optuna.samplers.nsgaii.BLXAlphaCrossover()
    sampler = mod.MOEADSampler(
        population_size=population_size,
        scalar_aggregation_func="tchebycheff",
        n_neighbors=population_size // 10,
        crossover=crossover,
    )
    study = optuna.create_study(
        sampler=sampler,
        study_name=f"{sampler.__class__.__name__}",
        directions=["minimize", "minimize"],
    )
    study.optimize(objective, n_trials=n_trials)

    optuna.visualization.plot_pareto_front(study).show()
