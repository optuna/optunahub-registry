import optuna
import optunahub


if __name__ == "__main__":

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", 0, 5)
        y = trial.suggest_float("y", 0, 3)
        v0 = 4 * x**2 + 4 * y**2
        v1 = (x - 5) ** 2 + (y - 5) ** 2
        return v0, v1

    samplers = [
        optunahub.load_local_module("samplers/mocma", registry_root="package").MoCmaSampler(
            popsize=100,
            seed=42,
        ),
        optuna.samplers.NSGAIISampler(population_size=100, seed=42),
    ]
    studies = []
    for sampler in samplers:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            sampler=sampler,
            study_name=f"{sampler.__class__.__name__}",
        )
        study.optimize(objective, n_trials=1000, n_jobs=1)
        studies.append(study)

    optunahub.load_module("visualization/plot_pareto_front_multi").plot_pareto_front(
        studies
    ).show()
    optunahub.load_module("visualization/plot_hypervolume_history_multi").plot_hypervolume_history(
        studies, reference_point=[200.0, 100.0]
    ).show()
