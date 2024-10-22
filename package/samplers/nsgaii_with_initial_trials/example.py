import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)
    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


storage = optuna.storages.InMemoryStorage()
# Sampling 0 generation using enqueueing & qmc sampler
study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.QMCSampler(seed=42),
    study_name="test",
    storage=storage,
)
study.enqueue_trial(
    {
        "x": 0,
        "y": 0,
    }
)
study.optimize(objective, n_trials=128)

# Using previous sampling results as the initial generation,
# sampled by NSGAII.
sampler = optunahub.load_module(
    "samplers/nsgaii_with_initial_trials",
).NSGAIIwITSampler(population_size=25, seed=42)

study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=sampler,
    study_name="test",
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)

optuna.visualization.plot_pareto_front(study).show()
