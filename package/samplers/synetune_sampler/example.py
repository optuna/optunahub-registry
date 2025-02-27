import optuna
import optunahub


module = optunahub.load_module("samplers/synetune_sampler")
SyneTuneSampler = module.syne_tune_sampler


if __name__ == "__main__":
    n_trials = 100

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return x**2 + y**2

    # Select method of choice here, i.e. Conformalized Quantile Regression (cqr)
    method_param = "cqr"
    sampler = SyneTuneSampler(
        {
            "x": optuna.distributions.FloatDistribution(-10, 10),
            "y": optuna.distributions.IntDistribution(-10, 10),
        },
        searcher_method=method_param,
        metric="mean_loss",
    )

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
