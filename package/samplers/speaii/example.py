import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)
    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


module = optunahub.load_module(
    "samplers/speaii",
)
mutation = module.PolynomialMutation(eta=20)
sampler = module.SPEAIISampler(population_size=50, archive_size=50, mutation=mutation)

study = optuna.create_study(
    sampler=sampler,
    directions=["minimize", "minimize"],
)
study.optimize(objective, n_trials=1000)

optuna.visualization.plot_pareto_front(study).show()
