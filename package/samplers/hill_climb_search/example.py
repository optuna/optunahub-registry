import optuna
import optunahub

if __name__ == "__main__":
    mod = optunahub.load_module("samplers/hill_climb_search")
    
    def objective(trial):
        x = trial.suggest_discrete_uniform("x", -10, 10)
        y = trial.suggest_discrete_uniform("y", -10, 10)
        return -(x**2 + y**2)

    sampler = mod.HillClimbSearch()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)

    print(study.best_trial.value, study.best_trial.params)
