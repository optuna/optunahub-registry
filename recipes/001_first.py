"""
.. _first:

How to implement and register your algorithm in OptunaHub
=========================================================

OptunaHub is a registry of third-party Optuna packages.
It provides a platform for users to share their own optimization algorithms and to use others' algorithms.
This recipe shows how to implement and register your own sampling algorithm in OptunaHub.

How to implement your own algorithm
-----------------------------------

Usually, Optuna provides `BaseSampler` class to implement your own sampler.
However, it is a bit complicated to implement a sampler from scratch.
Instead, in OptunaHub, you can use `samplers/simple/SimpleSampler` class, which is a sampler template that can be easily extended.

You need to install `optuna` to implement your own sampler, and `optunahub` to use the template `SimpleSampler`.

.. code-block:: bash

    $ pip install optuna optunahub

"""

###################################################################################################
# First of all, import `optuna`, `optunahub`, and other necessary modules.
from typing import Any

import numpy as np
import optuna
import optunahub


###################################################################################################
# Next, define your own sampler class by inheriting `SimpleSampler` class.
# In this example, we implement a sampler that always returns a random value.
# The `SimpleSampler` class can be loaded using `optunahub.load` function.
# The `force_load` argument is set to `True` to force loading the sampler without caching and consent to use stats.

SimpleSampler = optunahub.load(
    "samplers/simple",
).SimpleSampler


class MySampler(SimpleSampler):  # type: ignore
    # The `search_space` argument is necessary for the concrete implementation of the `SimpleSampler` class.
    def __init__(self, search_space: dict[str, optuna.distributions.BaseDistribution]) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()

    # You need to implement the `sample_relative` method.
    # This method returns a dictionary of hyperparameters.
    # The keys of the dictionary are the names of the hyperparameters, which must be the same as the keys of the `search_space` argument.
    # The values of the dictionary are the values of the hyperparameters.
    # In this example, the `sample_relative` method returns a dictionary of hyperparameters with random values.
    def sample_relative(
        self,
        study: optuna.study.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution],
    ) -> dict[str, Any]:
        # The `search_space` argument is exactly same as the `search_space` argument of the `__init__` method.
        # It is automatically handled by Optuna and `SimpleSampler`.

        params = {}
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
# In the following example, we create a study and optimize it using the `MySampler` class.
sampler = MySampler({"x": optuna.distributions.FloatDistribution(-10, 10)})
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)

###################################################################################################
# You can get the best parameter as follows.

best_params = study.best_params
found_x = best_params["x"]
print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))

###################################################################################################
# We can see that the ``x`` value found by Optuna is close to the optimal value of ``2``.

###################################################################################################
# How to register your implemented algorithm in OptunaHub
# -------------------------------------------------------
#
# After implementing your own algorithm, you can register it in OptunaHub.
# You need to create a pull request to the `optunahub-registry <https://github.com/optuna/optunahub-registry>`_ repository.
#
# The following is an example of the directory structure of the pull request.
#
# | package
# | └── samplers
# |     └── YOUR_ALGORITHM_NAME
# |         ├── README.md
# |         ├── __init__.py
# |         └── YOUR_ALGORITHM_NAME.py
#
# If you implement not visualizarion feature but sampler, you should put your implementation in the `samplers` directory.
# In the `samplers` directory, you should create a directory named after your algorithm.
# In the directory, you should put the following files:
#
# - `README.md`: A description of your algorithm. This file is used as the description of your algorithm in OptunaHub. See `package/samplers/simple/README.md <` for an example.
# - `__init__.py`: An initialization file. This file must import your impelemented sampler from `YOUR_ALGORITHM_NAME.py`.
# - `YOUR_ALGORITHM_NAME.py`: Your implemented sampler.
