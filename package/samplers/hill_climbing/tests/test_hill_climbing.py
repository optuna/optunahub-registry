"""Tests for the Hill Climbing Sampler."""

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
import optunahub
import pytest


# Load the module using optunahub
module = optunahub.load_local_module(package="samplers/hill_climbing", registry_root="package/")


class TestHillClimbingSampler:
    """Test cases for HillClimbingSampler."""

    def test_initialization(self) -> None:
        """Test sampler initialization with different parameters."""
        # Test default initialization
        sampler = module.HillClimbingSampler()
        assert sampler._neighbor_size == 5
        assert sampler._max_restarts == 10
        assert sampler._restart_count == 0
        assert not sampler._is_initialized

        # Test custom initialization
        sampler_custom = module.HillClimbingSampler(neighbor_size=10, max_restarts=20, seed=42)
        assert sampler_custom._neighbor_size == 10
        assert sampler_custom._max_restarts == 20

    def test_supported_distributions(self) -> None:
        """Test that only supported distributions are accepted."""
        # Supported distributions
        search_space_valid = {
            "x": IntDistribution(0, 10),
            "y": CategoricalDistribution(["A", "B", "C"]),
        }

        sampler = module.HillClimbingSampler()

        # This should not raise an error
        sampler._validate_search_space(search_space_valid)

        # Unsupported distribution should raise ValueError
        search_space_invalid = {
            "x": IntDistribution(0, 10),
            "z": FloatDistribution(0.0, 1.0),  # Not supported
        }

        with pytest.raises(
            ValueError, match="only supports IntDistribution and CategoricalDistribution"
        ):
            sampler._validate_search_space(search_space_invalid)

    def test_basic_optimization_minimize(self) -> None:
        """Test basic optimization with hill climbing sampler for minimization."""

        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_categorical("y", ["A", "B", "C"])
            penalty = {"A": 0, "B": 1, "C": 4}[y]
            return x**2 + penalty

        sampler = module.HillClimbingSampler(neighbor_size=3, max_restarts=2, seed=42)
        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(objective, n_trials=30)

        # Should find a reasonable solution
        assert study.best_value is not None
        assert study.best_value < 20  # Should be much better than random

    def test_basic_optimization_maximize(self) -> None:
        """Test basic optimization with hill climbing sampler for maximization."""

        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_categorical("y", ["A", "B", "C"])
            bonus = {"A": 10, "B": 5, "C": 0}[y]
            return -(x**2) + bonus  # Maximize negative quadratic + bonus

        sampler = module.HillClimbingSampler(neighbor_size=3, max_restarts=2, seed=42)
        study = optuna.create_study(sampler=sampler, direction="maximize")
        study.optimize(objective, n_trials=30)

        # Should find a reasonable solution (close to x=0 with y="A")
        assert study.best_value is not None
        assert study.best_value > 5  # Should find good solution

    def test_study_direction_support(self) -> None:
        """Test that the sampler correctly handles both minimize and maximize directions."""

        def simple_objective(trial: optuna.Trial) -> float:
            x = trial.suggest_int("x", 0, 10)
            return x

        # Test minimization
        sampler_min = module.HillClimbingSampler(seed=42)
        study_min = optuna.create_study(sampler=sampler_min, direction="minimize")
        study_min.optimize(simple_objective, n_trials=15)

        # Test maximization
        sampler_max = module.HillClimbingSampler(seed=42)
        study_max = optuna.create_study(sampler=sampler_max, direction="maximize")
        study_max.optimize(simple_objective, n_trials=15)

        # For minimization, should prefer lower values
        # For maximization, should prefer higher values
        assert study_min.best_value is not None
        assert study_max.best_value is not None
        assert study_min.best_value <= study_max.best_value


if __name__ == "__main__":
    # Run tests if file is executed directly
    pytest.main([__file__])
