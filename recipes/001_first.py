"""
.. _first:

How to Implement Your Sampler with OptunaHub
===========================================================

OptunaHub is an Optuna package registry, which is a platform to share algorithms developed by contributors.
This recipe shows how to implement your own algorithm with OptunaHub.

Here, we show how to implement your own sampler, i.e., optimizaiton algorithm.
If you want to implement algorithms other than a sampler, please refer to the other recipes.

- :doc:`003_pruner`
- :doc:`004_visualization`

Usually, Optuna provides `BaseSampler` class to implement your own sampler.
However, it is a bit complicated to implement a sampler from scratch.
Instead, in OptunaHub, you can use `samplers/simple/SimpleBaseSampler` class, which is a sampler template that can be easily extended.

You need to install `optuna` to implement your own sampler, and `optunahub` to use the template `SimpleBaseSampler`.

.. code-block:: bash

    $ pip install optuna optunahub

"""

###################################################################################################
# First of all, import `optuna`, `optunahub`, and other required modules.
from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import optunahub


###################################################################################################
# Next, define your own sampler class by inheriting `SimpleBaseSampler` class.
# In this example, we implement a sampler that returns a random value.
# `SimpleBaseSampler` class can be loaded using `optunahub.load_module` function.
# `force_reload=True` argument forces downloading the sampler from the registry.
# If we set `force_reload` to `False`, we use the cached data in our local storage if available.

SimpleBaseSampler = optunahub.load_module("samplers/simple").SimpleBaseSampler


class MySampler(SimpleBaseSampler):  # type: ignore
    # By default, search space will be estimated automatically like Optuna's built-in samplers.
    # You can fix the search spacd by `search_space` argument of `SimpleBaseSampler` class.
    def __init__(
        self, search_space: dict[str, optuna.distributions.BaseDistribution] | None = None
    ) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()

    # You need to implement `sample_relative` method.
    # This method returns a dictionary of hyperparameters.
    # The keys of the dictionary are the names of the hyperparameters, which must be the same as the keys of the `search_space` argument.
    # The values of the dictionary are the values of the hyperparameters.
    # In this example, `sample_relative` method returns a dictionary of randomly sampled hyperparameters.
    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        # `search_space` argument must be identical to `search_space` argument input to `__init__` method.
        # This method is automatically invoked by Optuna and `SimpleBaseSampler`.

        # If search space is empty, all parameter values are sampled randomly by SimpleBaseSampler.
        if search_space == {}:
            return {}

        params = {}  # type: dict[str, Any]
        for n, d in search_space.items():
            if isinstance(d, optuna.distributions.FloatDistribution):
                params[n] = self._rng.uniform(d.low, d.high)
            elif isinstance(d, optuna.distributions.IntDistribution):
                params[n] = self._rng.randint(d.low, d.high)
            elif isinstance(d, optuna.distributions.CategoricalDistribution):
                params[n] = d.choices[0]
            else:
                raise NotImplementedError
        return params


###################################################################################################
# In this example, the objective function is a simple quadratic function.


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return x**2


###################################################################################################
# This sampler can be used in the same way as other Optuna samplers.
# In the following example, we create a study and optimize it using `MySampler` class.
sampler = MySampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)

###################################################################################################
# The best parameter can be fetched as follows.

best_params = study.best_params
found_x = best_params["x"]
print(f"Found x: {found_x}, (x - 2)^2: {(found_x - 2) ** 2}")

###################################################################################################
# We can see that ``x`` value found by Optuna is close to the optimal value ``2``.
#
# In the next recipe, we will show how to register your sampler to OptunaHub.
# Let's move on to :doc:`002_registration`.


###################################################################################################
# In the above examples, search space is estimated at the first trial and updated dynamically through optimization.
# If you pass the search space to the sampler, you can avoid the overhead of estimating the search space.
sampler = MySampler({"x": optuna.distributioins.FloatDictribution(-10, 10)})
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
