import optuna
import optunahub

if __name__ == "__main__":    
    def objective(trial):
        x = trial.suggest_discrete_uniform("x", -10, 10)
        y = trial.suggest_discrete_uniform("y", -10, 10)
        return -(x**2 + y**2)

    module = optunahub.load_module(
        package="samplers/hill-climb-search",
        repo_owner="csking101",
        ref="hill-climb-algorithm")
    sampler = module.HillClimbSearch()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)

    print(study.best_trial.value, study.best_trial.params)
