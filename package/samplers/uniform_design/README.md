---
author: Yotaro Yamaguchi
title: Uniform Design Sampler
description: A sampler based on the uniform design algorithm.
tags: [sampler, design of experiments]
optuna_versions: [3.6.1.]
license: MIT License
---

## Abstract

This package provides an implementation of the uniform design (UD) algorithm.
UD is a variety of design-of-experiment (DOE) methods, and it has better sample efficiency than simple random sampling.

## Class or Function Names

- UniformDesignSampler

## Installation

```shell
$ pip install -r https://hub.optuna.org/samplers/uniform_design/requirements.txt
```

## Example

```python
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optunahub
from optuna.distributions import FloatDistribution


module = optunahub.load_module("samplers/uniform_design")
UniformDesignSampler = module.UniformDesignSampler



def objective(trial):
    x = trial.suggest_float("x", 0, 1)
    y = trial.suggest_float("y", 0, 1)
    obj = 2 * np.cos(10 * x) * np.sin(10 * y) + np.sin(10 * x * y)
    return obj


def objective_show(parameters):
    x1 = parameters["x"]
    x2 = parameters["y"]
    obj = 2 * np.cos(10 * x1) * np.sin(10 * x2) + np.sin(10 * x1 * x2)
    return obj


# Define the search space
search_space = {"x": FloatDistribution(0, 1), "y": FloatDistribution(0, 1)}

# Create the study
discretization_level = 20
sampler = UniformDesignSampler(search_space, discretization_level)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=40, n_jobs=2)

logs = study.trials_dataframe()


def plot_trajectory(xlim, ylim, func, logs, title):
    grid_num = 25
    xlist = np.linspace(xlim[0], xlim[1], grid_num)
    ylist = np.linspace(ylim[0], ylim[1], grid_num)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.zeros((grid_num, grid_num))
    for i, x in enumerate(xlist):
        for j, y in enumerate(ylist):
            Z[j, i] = func({"x": x, "y": y})

    cp = plt.contourf(X, Y, Z)
    plt.scatter(logs.loc[:, ["params_x"]], logs.loc[:, ["params_y"]], color="red")
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.colorbar(cp)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)


plot_trajectory([0, 1], [0, 1], objective_show, logs, "UD")
plt.show()
```

## Others

### Reference

Kai-Tai Fang, Dennis KJ Lin, Peter Winker, and Yong Zhang. Uniform design: theory and
application. Technometrics, 42(3):237–248, 2000.
