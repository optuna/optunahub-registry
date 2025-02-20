import optuna
import optunahub
from syne_tune.blackbox_repository import load_blackbox


module = optunahub.load_module("samplers/synetune_sampler")
SyneTuneSampler = module.SyneTuneSampler

if __name__ == "__main__":
    n_trials = 100
    # Load a black-box benchmark (e.g., nasbench201)
    blackbox = load_blackbox("nasbench201")["cifar100"]
    config_space = blackbox.configuration_space

    def objective(trial: optuna.trial.Trial) -> float:
        config = {
            "hp_x0": trial.suggest_categorical("hp_x0", config_space["hp_x0"]),
            "hp_x1": trial.suggest_categorical("hp_x1", config_space["hp_x1"]),
            "hp_x2": trial.suggest_categorical("hp_x2", config_space["hp_x2"]),
            "hp_x3": trial.suggest_categorical("hp_x3", config_space["hp_x3"]),
            "hp_x4": trial.suggest_categorical("hp_x4", config_space["hp_x4"]),
            "hp_x5": trial.suggest_categorical("hp_x5", config_space["hp_x5"]),
        }
        result = blackbox.objective_function(config)
        return float(result[-1][0])

    # TODO sanity check mit syne tune, sieht es gleich aus??
    # Select method of choice here, i.e. Conformalized Quantile Regression (cqr)
    method_param = "cqr"
    sampler = SyneTuneSampler(
        {
            "hp_x0": optuna.distributions.CategoricalDistribution(config_space["hp_x0"]),
            "hp_x1": optuna.distributions.CategoricalDistribution(config_space["hp_x1"]),
            "hp_x2": optuna.distributions.CategoricalDistribution(config_space["hp_x2"]),
            "hp_x3": optuna.distributions.CategoricalDistribution(config_space["hp_x3"]),
            "hp_x4": optuna.distributions.CategoricalDistribution(config_space["hp_x4"]),
            "hp_x5": optuna.distributions.CategoricalDistribution(config_space["hp_x5"]),
        },
        searcher_method=method_param,
        metric="mean_loss",
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
