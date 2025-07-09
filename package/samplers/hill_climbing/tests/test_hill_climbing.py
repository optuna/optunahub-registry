"""Tests for the Hill Climbing Sampler."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import optuna
from optuna.distributions import IntDistribution, CategoricalDistribution, FloatDistribution

# Import directly from the parent directory
from _sampler import HillClimbingSampler


class TestHillClimbingSampler:
    """Test cases for HillClimbingSampler."""
    
    def test_initialization(self):
        """Test sampler initialization with different parameters."""
        # Test default initialization
        sampler = HillClimbingSampler()
        assert sampler._neighbor_size == 5
        assert sampler._max_restarts == 10
        assert sampler._restart_count == 0
        assert not sampler._is_initialized
        
        # Test custom initialization
        sampler_custom = HillClimbingSampler(
            neighbor_size=10,
            max_restarts=20,
            seed=42
        )
        assert sampler_custom._neighbor_size == 10
        assert sampler_custom._max_restarts == 20
    
    def test_supported_distributions(self):
        """Test that only supported distributions are accepted."""
        # Supported distributions
        search_space_valid = {
            "x": IntDistribution(0, 10),
            "y": CategoricalDistribution(["A", "B", "C"])
        }
        
        sampler = HillClimbingSampler()
        
        # This should not raise an error
        sampler._validate_search_space(search_space_valid)
        
        # Unsupported distribution should raise ValueError
        search_space_invalid = {
            "x": IntDistribution(0, 10),
            "z": FloatDistribution(0.0, 1.0)  # Not supported
        }
        
        with pytest.raises(ValueError, match="only supports IntDistribution and CategoricalDistribution"):
            sampler._validate_search_space(search_space_invalid)
    
    def test_basic_optimization(self):
        """Test basic optimization with hill climbing sampler."""
        def objective(trial):
            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_categorical("y", ["A", "B", "C"])
            penalty = {"A": 0, "B": 1, "C": 4}[y]
            return x**2 + penalty
        
        sampler = HillClimbingSampler(neighbor_size=3, max_restarts=2, seed=42)
        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(objective, n_trials=30)
        
        # Should find a reasonable solution
        assert study.best_value is not None
        assert study.best_value < 20  # Should be much better than random


if __name__ == "__main__":
    # Run tests if file is executed directly
    pytest.main([__file__])