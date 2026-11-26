"""OptunaHub wrapper for the steady-state genetic algorithm sampler in strategy-genome.

The implementation lives in the strategy-genome package on PyPI
(https://pypi.org/project/strategy-genome/, source at
https://github.com/Mattbusel/strategy-genome), so fixes ship through normal releases.
This module only re-exports the sampler for ``optunahub.load_module``.
"""

from __future__ import annotations

from strategy_genome.optuna import GeneticStrategySampler


__all__ = ["GeneticStrategySampler"]
