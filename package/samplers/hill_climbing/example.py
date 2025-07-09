"""Example usage of the Hill Climbing Sampler.

This example demonstrates how to use the HillClimbingSampler for discrete optimization
problems with both integer and categorical parameters.
"""

import optuna
import optunahub


def discrete_optimization_objective(trial: optuna.Trial) -> float:
    """Objective function for discrete optimization.

    This function optimizes a combination of integer and categorical parameters.
    The goal is to minimize the objective value.
    """
    # Integer parameters
    x = trial.suggest_int("x", -20, 20)
    y = trial.suggest_int("y", -15, 15)

    # Categorical parameters
    algorithm = trial.suggest_categorical("algorithm", ["method_A", "method_B", "method_C"])
    preprocessing = trial.suggest_categorical("preprocessing", ["norm", "scale", "none"])

    # Define penalties for categorical choices
    algorithm_penalty = {
        "method_A": 0,  # Best choice
        "method_B": 10,  # Medium choice
        "method_C": 25,  # Worst choice
    }

    preprocessing_bonus = {
        "norm": -5,  # Bonus for normalization
        "scale": -2,  # Small bonus for scaling
        "none": 0,  # No bonus
    }

    # Calculate objective value
    # Minimize the sum of squares plus categorical penalties/bonuses
    objective_value = (
        x**2 + y**2 + algorithm_penalty[algorithm] + preprocessing_bonus[preprocessing]
    )

    return objective_value


def knapsack_like_objective(trial: optuna.Trial) -> float:
    """Knapsack-like optimization problem.

    Select items with different weights and values to maximize value
    while staying under weight constraint.
    """
    # Item selections (binary choices represented as categorical)
    items = {}
    for i in range(8):
        items[f"item_{i}"] = trial.suggest_categorical(f"item_{i}", [0, 1])

    # Item properties (weight, value)
    item_properties = {
        "item_0": (2, 3),  # weight=2, value=3
        "item_1": (3, 4),  # weight=3, value=4
        "item_2": (4, 5),  # weight=4, value=5
        "item_3": (5, 6),  # weight=5, value=6
        "item_4": (1, 1),  # weight=1, value=1
        "item_5": (6, 9),  # weight=6, value=9
        "item_6": (7, 10),  # weight=7, value=10
        "item_7": (3, 5),  # weight=3, value=5
    }

    total_weight = sum(item_properties[item][0] * selected for item, selected in items.items())
    total_value = sum(item_properties[item][1] * selected for item, selected in items.items())

    # Weight constraint
    max_weight = 15

    # Penalty for exceeding weight limit
    if total_weight > max_weight:
        penalty = (total_weight - max_weight) * 100
        return -total_value + penalty

    # Minimize negative value (equivalent to maximizing value)
    return -total_value


def main() -> None:
    """Run hill climbing optimization examples."""

    print("Hill Climbing Sampler Examples")
    print("=" * 40)

    # Load the hill climbing sampler
    module = optunahub.load_module("samplers/hill_climbing")
    HillClimbingSampler = module.HillClimbingSampler

    # Example 1: Discrete optimization with mixed parameter types
    print("\nExample 1: Mixed Integer and Categorical Optimization")
    print("-" * 50)

    sampler1 = HillClimbingSampler(neighbor_size=6, max_restarts=3, seed=42)

    study1 = optuna.create_study(sampler=sampler1, direction="minimize")
    study1.optimize(discrete_optimization_objective, n_trials=50)

    print(f"Best value: {study1.best_value}")
    print(f"Best parameters: {study1.best_params}")
    print(f"Number of trials: {len(study1.trials)}")

    # Example 2: Knapsack-like problem
    print("\nExample 2: Knapsack-like Binary Optimization")
    print("-" * 50)

    sampler2 = HillClimbingSampler(neighbor_size=8, max_restarts=5, seed=123)

    study2 = optuna.create_study(sampler=sampler2, direction="minimize")
    study2.optimize(knapsack_like_objective, n_trials=75)

    print(f"Best value: {study2.best_value}")
    print(f"Best parameters: {study2.best_params}")
    print(f"Number of trials: {len(study2.trials)}")

    # Calculate actual knapsack solution
    selected_items = [item for item, selected in study2.best_params.items() if selected == 1]
    print(f"Selected items: {selected_items}")

    # Example 3: Demonstrating restart mechanism
    print("\nExample 3: Demonstrating Restart Mechanism")
    print("-" * 50)

    def challenging_objective(trial: optuna.Trial) -> float:
        """Objective with multiple local optima to demonstrate restarts."""
        x = trial.suggest_int("x", 0, 100)
        y = trial.suggest_int("y", 0, 100)

        # Create multiple local optima
        if 10 <= x <= 20 and 10 <= y <= 20:
            return (x - 15) ** 2 + (y - 15) ** 2  # Local optimum around (15, 15)
        elif 70 <= x <= 80 and 70 <= y <= 80:
            return (x - 75) ** 2 + (y - 75) ** 2 - 50  # Better optimum around (75, 75)
        else:
            return x**2 + y**2 + 100  # High penalty elsewhere

    sampler3 = HillClimbingSampler(
        neighbor_size=4,
        max_restarts=8,  # More restarts to escape local optima
        seed=456,
    )

    study3 = optuna.create_study(sampler=sampler3, direction="minimize")
    study3.optimize(challenging_objective, n_trials=100)

    print(f"Best value: {study3.best_value}")
    print(f"Best parameters: {study3.best_params}")
    print(f"Number of trials: {len(study3.trials)}")

    print("\nOptimization completed!")


if __name__ == "__main__":
    main()
