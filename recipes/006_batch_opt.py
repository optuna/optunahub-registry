"""
.. _batch_opt:

How to Wrap a Sampler for Batch Optimization with OptunaHub
===========================================================

Since Optuna usually assumes asynchronous optimization methods, the registration of batch optimization methods is not so straightforward.
In this tutorial, we describe how to wrap a batch optimization method.

Please note that we also have the tutorial for the sampler registration as well:

- :doc:`001_first`

As mentioned in the tutorial above, samplers in Optuna usually need to inherit ``BaseSampler`` class.
On the other hand, as it is not really simple to wrap a third-party sampler in the Optuna manner, we instead provide `optunahub.samplers.SimpleBaseSampler <https://optuna.github.io/optunahub/generated/optunahub.samplers.SimpleBaseSampler.html>`__ class to make a wrapping easier.

To implement your own sampler, ``optuna`` and ``optunahub`` must be installed:

.. code-block:: bash

    $ pip install optuna optunahub

"""

###################################################################################################
# We first import all the modules necessary for this tutorial.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import optuna
from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub


###################################################################################################
# In this tutorial, we explain how to wrap an external optimization library with the batch optimization interface.
# To do so, we first define a batch optimization sampler.
# In this example, we consider the simplest example where the sampler optimizes continuous parameters each with the range between -1e5 and 1e5.
class SamplerToWrap:
    def __init__(self, dim: int, batch_size: int, seed: int | None) -> None:
        self._batch_size = batch_size
        self._dim = dim
        self._rng = np.random.RandomState(seed)
        self._mean = np.zeros(dim, dtype=float)
        self._cov = np.identity(dim, dtype=float) * 1e9

    def ask(self) -> np.ndarray:
        # Return a batch of parameters.
        return self._rng.multivariate_normal(mean=self._mean, cov=self._cov, size=self._batch_size)

    def tell(self, params: np.ndarray, values: np.ndarray) -> None:
        # Report a batch of parameters and the corresponding value.
        assert len(params) == len(values) and params.shape[-1] == self._dim
        # Take quantile so that at least two solutions will be considered as good.
        quantile = max(2 / len(values), 0.1)
        good_value = np.quantile(values, quantile)
        good_params = params[values <= good_value]
        # Take the statistics of good parameters.
        self._mean = np.mean(good_params, axis=0)
        self._cov = np.cov(good_params, rowvar=False)
        print(good_params)
        print(self._cov)


###################################################################################################
# We now show how to wrap the sampler above to adapt to the Optuna interface.
# Please note that we can guarantee the correctness of this implementation only if no trial crashes and the study is a non-parallel setup.
# Most importantly, users need to modify only the initialization constructor, ``_ask_wrapper``, and ``_tell_wrapper`` below:
class BatchOptSampler(optunahub.samplers.SimpleBaseSampler):
    def __init__(
        self,
        # The arguments must be adapted according to the sampler to be wrapped.
        search_space: dict[str, BaseDistribution],
        batch_size: int,
        seed: int | None = None,
    ) -> None:
        # This implementation does not support parallel optimization.
        self._external_sampler = SamplerToWrap(
            batch_size=batch_size, dim=len(search_space), seed=seed
        )
        self._batch_size = batch_size
        self._param_names = list(search_space.keys())
        super().__init__(search_space=search_space)

        # Store the batch results.
        self._params_to_tell: list[np.ndarray] = []
        self._values_to_tell: list[np.ndarray] = []

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, float]:
        # Ask the next batch.
        params_batch = self._ask_wrapper()

        # Use the first entry as the next parameter.
        next_params = params_batch[0]

        # Enqueue the parameters except for the first one.
        for params in params_batch[1:]:
            study.enqueue_trial(params)

        return next_params

    def _ask_wrapper(self) -> list[dict[str, Any]]:
        # NOTE: Contributors need to modify here accordingly.
        params_array = self._external_sampler.ask()

        # Convert every entry into the Optuna format and return them as a list.
        return [
            {name: params[d] for d, name in enumerate(self._param_names)}
            for params in params_array
        ]

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        assert values is not None, "This implementation assumes no trial crashes."
        # Store the trial result.
        self._params_to_tell.append([trial.params[name] for name in self._param_names])
        self._values_to_tell.append(values[0])

        if len(self._values_to_tell) != self._batch_size:
            return

        # Tell the batch results to external sampler once the batch is ready.
        self._tell_wrapper()

        # Empty the results.
        self._params_to_tell = []
        self._values_to_tell = []

    def _tell_wrapper(self) -> None:
        # NOTE: Contributors need to modify here accordingly.
        self._external_sampler.tell(
            params=np.asarray(self._params_to_tell), values=np.asarray(self._values_to_tell)
        )


###################################################################################################
# Given the sampler above, let's optimize an objective function below:
def objective(trial: optuna.Trial) -> float:
    x0 = trial.suggest_float("x0", -1e5, 1e5)
    x1 = trial.suggest_float("x1", -1e5, 1e5)
    return x0**2 + x1**2


###################################################################################################
# To optimize the function with the wrapped sampler, we need to use the following code:
search_space = {
    "x0": optuna.distributions.FloatDistribution(-1e5, 1e5),
    "x1": optuna.distributions.FloatDistribution(-1e5, 1e5),
}
sampler = BatchOptSampler(search_space=search_space, batch_size=100)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=3000)

###################################################################################################
# The best parameters can be found as follows.
best_params = study.best_params
best_value = study.best_value
print(f"Best params: {best_params}, Best value: {best_value}")

###################################################################################################
# As expected, ``best_params`` takes the values near the optimum, which is ``{"x0":0, "x1": 0}``.
