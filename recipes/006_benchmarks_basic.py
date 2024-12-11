"""
.. _benchmarks_basic:

How to Implement Your Benchmark Problems with OptunaHub (Basic)
===============================================================

OptunaHub provides the ``optunahub.benchmarks`` module for implementing benchmark problems.
In this tutorial, we will explain how to implement your own benchmark problems using ``optunahub.benchmarks``.
"""

###################################################################################################
# First of all, import `optuna` and other required modules.
from __future__ import annotations

import optuna
from optunahub.benchmarks import BaseProblem


###################################################################################################
# Next, define your own problem class by inheriting ``BaseProblem`` class.
# Here, let's implement a simple 2-dimensional sphere function.
#
# You need to implement the following methods defined in the ``BaseProblem`` class.:
#
# - ``search_space``: This method returns the dictionary of search space of the problem. Each dictionary element consists of the parameter name and distribution (see `optuna.distributions <https://optuna.readthedocs.io/en/stable/reference/distributions.html>`__).
# - ``directions``: This method returns the directions of the problem. The return type is the list of `optuna.study.direction <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.StudyDirection.html>`__.
# - ``evaluate``: This method evaluates the objective function by taking the dictionary of input parameters.
class Sphere2D(BaseProblem):
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            "x0": optuna.distributions.FloatDistribution(low=-5, high=5),
            "x1": optuna.distributions.FloatDistribution(low=-5, high=5),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MINIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        return params["x0"] ** 2 + params["x1"] ** 2


###################################################################################################
# Since ``BaseProblem`` provides the default implementation of ``__call__(optuna.Trial)`` that calls the ``evaluate`` method with the parameters defined in the search space, you can use the problem instance as an objective function for ``study.optimize``.
sphere2d = Sphere2D()
study = optuna.create_study(directions=sphere2d.directions)
study.optimize(sphere2d, n_trials=20)


###################################################################################################
# You can also customize the constructor of the problem class and introduce additional attributes.
# The below example implements a variadic-dimensional sphere function.
class SphereND(BaseProblem):
    def __init__(self, dim: int) -> None:
        self.dim = dim

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        return {
            f"x{i}": optuna.distributions.FloatDistribution(low=-5, high=5)
            for i in range(self.dim)
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MINIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        return sum(params[f"x{i}"] ** 2 for i in range(self.dim))


sphere3d = SphereND(dim=3)
study2 = optuna.create_study(directions=sphere3d.directions)
study2.optimize(sphere3d, n_trials=20)

###################################################################################################
# After implementing your own pruner, you can register it with OptunaHub.
# See :doc:`002_registration` for how to register your pruner with OptunaHub.
#
# In :ref:`benchmarks_advanced`, how to implement complex benchmark problems such as a problem with dynamic search space are explained.
