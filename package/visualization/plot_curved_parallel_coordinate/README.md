---
author: Rishabh Dewangan
title: Curved Parallel Coordinates
description: A parallel coordinate plot with smooth, monotonic curves to reduce visual clutter.
tags: [visualization, plotly, parallel-coordinates, explainable-ai]
optuna_versions: [4.7.0]
license: MIT License
---

## Abstract

This package provides an Optuna parallel coordinate plot using Piecewise Cubic Hermite Interpolating Polynomials (Pchip) for smooth, monotonic curves. This reduces visual clutter and makes individual trial trajectories easier to track in high-dimensional spaces compared to standard straight-line plots.

## APIs

- `plot_curved_parallel_coordinate(study: optuna.Study, params: list[str] | None = None, points_per_segment: int = 50) -> plotly.graph_objects.Figure`
  - `study`: The `optuna.Study` object (plots completed trials only).
  - `params`: List of parameter names to plot. Defaults to all parameters.
  - `points_per_segment`: Curve resolution. Defaults to `50`.

## Installation

This package relies on standard mathematical and visualization libraries.

```shell
$ pip install scipy plotly numpy optuna
```

## Example

```python
import optuna
import optunahub

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    z = trial.suggest_float("z", -5, 5)
    return x**2 + y**2 + z**2

study = optuna.create_study()
study.optimize(objective, n_trials=30)

# Load the curved parallel coordinate module from OptunaHub
module = optunahub.load_module(package="visualization/plot_curved_parallel_coordinate")

# Generate and display the plot
fig = module.plot_curved_parallel_coordinate(study)
fig.show()
```
