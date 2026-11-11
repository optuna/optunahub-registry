import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)

    # Values that are not parameters can be saved as user attributes.
    trial.set_user_attr("distance", abs(x - y))
    trial.set_user_attr("is_x_positive", x > 0)
    return x**2 + y**2


study = optuna.create_study()
study.optimize(objective, n_trials=50)

m = optunahub.load_module("visualization/plot_slice_with_user_attrs")
fig = m.plot_slice(study, user_attrs=["distance", "is_x_positive"])
fig.show()
