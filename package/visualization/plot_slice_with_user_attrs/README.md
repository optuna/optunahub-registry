---
author: Atsushi Hidaka
title: Slice Plot with User Attributes
description: Plot the relationship between parameters, user attributes, and the objective value as slice plot in a study.
tags: [visualization, slice plot, user attributes]
optuna_versions: [4.9.0, 5.0.0]
license: MIT License
---

## Abstract

This package extends [`optuna.visualization.plot_slice`](https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_slice.html) so that it can also plot user attributes set by `trial.set_user_attr`.
User attributes are useful for recording values that are not parameters, such as training time, the number of model parameters, or whether early stopping happened.
With this package, you can check how such values relate to the objective value in the same figure as the parameters.

Without the `user_attrs` argument, the output is the same as `optuna.visualization.plot_slice`.

## APIs

- `plot_slice(study, params=None, *, user_attrs=None, target=None, target_name="Objective Value")`
  - `study`: A study whose trials are plotted for their target values.
  - `params`: Parameter list to visualize. The default is all parameters. Pass `[]` to plot only user attributes.
  - `user_attrs`: User attribute keys to visualize. The default is `None`, which plots no user attributes.
  - `target`: A function to specify the value to display. Specify this argument if `study` is being used for multi-objective optimization.
  - `target_name`: Target's name to display on the axis label.
  - Returns a `plotly.graph_objects.Figure`.

Subplots are ordered as parameters (sorted by name) and then user attributes (sorted by key).
The x-axis title of a user attribute is `user_attrs.` followed by the key, e.g., `user_attrs.distance`.

Each user attribute value is plotted as follows:

| Value                                                               | How it is plotted            |
| ------------------------------------------------------------------- | ---------------------------- |
| Missing key or `None`                                               | Not plotted (no warning)     |
| `bool`                                                              | Category (`True` / `False`)  |
| Finite `int` / `float` (including NumPy numbers such as `np.int64`) | Number on a linear axis      |
| `nan` / `inf`                                                       | Not plotted (with a warning) |
| `str`                                                               | Category                     |
| Others such as `list` and `dict`                                    | Not plotted (with a warning) |

A key is plotted on a numerical axis only when all of its values are numbers.
Otherwise, all values of the key are plotted as categories in the order of their first appearance.

Warnings are shown once per key and reason, instead of once per trial.
They follow `optuna.logging.set_verbosity`.

## Installation

```shell
pip install -r https://hub.optuna.org/visualization/plot_slice_with_user_attrs/requirements.txt
```

## Example

```python
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
```

## Others

### Notes

- `True`, `1`, and `1.0` are treated as different categories when a key has mixed types, because Python treats them as equal (`True == 1 == 1.0`).
- NumPy numbers such as `np.int64` can be stored as user attributes only with the in-memory storage. Other storages such as `RDBStorage` require JSON-serializable values.
- Infeasible trials of categorical values are also plotted in gray. (`optuna.visualization.plot_slice` in Optuna 4.9.0 and 5.0.0 does not plot them for categorical parameters.)

### Credits

The drawing code is copied and modified from [`optuna/visualization/_slice.py`](https://github.com/optuna/optuna/blob/master/optuna/visualization/_slice.py) in Optuna, which is distributed under the MIT License.
