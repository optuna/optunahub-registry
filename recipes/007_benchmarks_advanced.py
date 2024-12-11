"""
.. _benchmarks_advanced:

How to Implement Your Benchmark Problems with OptunaHub (Advanced)
==================================================================

OptunaHub provides the ``optunahub.benchmarks`` module for implementing benchmark problems.
In this tutorial, we will explain how to implement complex benchmark problems such as a problem with dynamic search space using ``optunahub.benchmarks``.

For the implementation of simple benchmark problems, please refer to :ref:`benchmarks_basic`.
"""

###################################################################################################
# Implementing a Problem with Dynamic Search Space
# -------------------------------------------------
#
# Here, let's implement a problem with a dynamic search space.
#
# First of all, import `optuna` and other required modules.
from __future__ import annotations

import optuna
from optunahub.benchmarks import BaseProblem


###################################################################################################
# Next, define your own problem class by inheriting ``BaseProblem`` class.
# To implement a problem with a dynamic search space, you should override the ``__call__(optuna.Trial)`` method that allows you to define the search space in a define-by-run manner.
# You also need to implement the ``direcitons`` property.
class DynamicProblem(BaseProblem):
    def __call__(self, trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -5, 5)
        if x < 0:
            # Parameter `y` exists only when `x` is negative.
            y = trial.suggest_float("y", -5, 5)
            return x**2 + y
        else:
            return x**2

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        return [optuna.study.StudyDirection.MINIMIZE]

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        # You can implement this property as you like, or leave it unimplemented (``BaseProblem`` provides this default behavior).
        raise NotImplementedError

    def evaluate(self, params: dict[str, float]) -> float:
        # You can implement this method as you like, or leave it unimplemented (``BaseProblem`` provides this default behavior).
        raise NotImplementedError


###################################################################################################
# The implementations of the ``search_space`` and ``evaluate`` are non-trivial when the search space is dynamic.
# However, the ``__call__(optuna.Trial)`` required to run Optuna optimizations no longer depends on the ``evaluate`` method or the ``search_space`` attribute, so you can implement it however you like, or even leave it unimplemented (then, calling them will result in a ``NotImplementedError``).

###################################################################################################
# Then, you can optimize the problem with Optuna as usual.
dynamic_problem = DynamicProblem()
study = optuna.create_study(directions=dynamic_problem.directions)
study.optimize(dynamic_problem, n_trials=20)

###################################################################################################
# After implementing your own benchmark problem, you can register it with OptunaHub.
# See :doc:`002_registration` for how to register your benchmark problem with OptunaHub.
