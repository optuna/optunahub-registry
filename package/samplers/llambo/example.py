"""
To run this example, you need to install the following packages:

matplotlib>=3.9.2
numpy>=2.1.3
scipy>=1.14.1
scikit-learn>=1.5.2

"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optunahub
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------Objective Functions---------------


def objective_Ackley(trial: optuna.Trial) -> float:
    """Ackley function optimization.

    Args:
        trial:
            The trial object to suggest parameters.

    Returns:
        The computed Ackley function value.
    """
    n_dimensions = 10  # High-dimensional problem with 10 dimensions

    # Suggest a value for each dimension
    variables = [trial.suggest_float(f"x{i}", -32.768, 32.768) for i in range(n_dimensions)]

    # Ackley function parameters
    a = 20
    b = 0.2
    c = 2 * math.pi

    # Compute the function components
    sum_sq_term = sum(x**2 for x in variables)
    cos_term = sum(math.cos(c * x) for x in variables)

    # Return the function value
    return (
        -a * math.exp(-b * math.sqrt(sum_sq_term / n_dimensions))
        - math.exp(cos_term / n_dimensions)
        + a
        + math.exp(1)
    )


def objective_sphere(trial: optuna.Trial) -> float:
    """Sphere function optimization.

    Args:
        trial:
            The trial object to suggest parameters.

    Returns:
        The computed Sphere function value.
    """
    n_dimensions = 10  # Number of dimensions

    # Suggest a value for each dimension
    variables = [trial.suggest_float(f"x{i}", -10.0, 10.0) for i in range(n_dimensions)]

    # Return the sum of squares
    return sum(x**2 for x in variables)


def objective_Rastrigin(trial: optuna.Trial) -> float:
    """Rastrigin function optimization.

    Args:
        trial:
            The trial object to suggest parameters.

    Returns:
        The computed Rastrigin function value.
    """
    n_dimensions = 10  # High-dimensional problem with 10 dimensions
    variables = [trial.suggest_float(f"x{i}", -5.12, 5.12) for i in range(n_dimensions)]
    A = 10

    # Compute the Rastrigin function value
    sum_term = sum(x**2 - A * math.cos(2 * math.pi * x) for x in variables)
    return A * n_dimensions + sum_term


def objective_Schwefel(trial: optuna.Trial) -> float:
    """Schwefel function optimization.

    Args:
        trial:
            The trial object to suggest parameters.

    Returns:
        The computed Schwefel function value.
    """
    n_dimensions = 10
    variables = [trial.suggest_float(f"x{i}", -500, 500) for i in range(n_dimensions)]

    # Compute the Schwefel function value
    sum_term = sum(x * math.sin(math.sqrt(abs(x))) for x in variables)
    return -(418.9829 * n_dimensions - sum_term) + 10000


def objective_ML(trial: optuna.Trial) -> float:
    """Machine learning objective using RandomForestClassifier.

    Args:
        trial:
            The trial object to suggest hyperparameters.

    Returns:
        Mean accuracy obtained using cross-validation.
    """
    # Load dataset
    data = load_digits()
    X, y = data.data, data.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter suggestions
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])

    # Define a pipeline with scaling and RandomForestClassifier
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    bootstrap=bootstrap,
                    random_state=42,
                ),
            ),
        ]
    )

    # Cross-validation for accuracy
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()


# ruff: noqa
def objective_dynamic_1(trial: optuna.Trial) -> float:
    """Dynamic search space function with a single additional parameter for the first trial.

    Args:
        trial:
            The trial object to suggest parameters.

    Returns:
        Computed value of the function.
    """
    x = trial.suggest_float("x", -5.12, 5.12)
    y = trial.suggest_float("y", -5.12, 5.12)
    if trial.number == 0:
        z = trial.suggest_float("z", -5.12, 5.12)
    return x**2 + y**2


def objective_dynamic_2(trial: optuna.Trial) -> float:
    """Dynamic search space function with a single additional parameter at the 100th trial.

    Args:
        trial:
            The trial object to suggest parameters.

    Returns:
        Computed value of the function.
    """
    x = trial.suggest_float("x", -5.12, 5.12)
    y = trial.suggest_float("y", -5.12, 5.12)
    if trial.number == 100:
        z = trial.suggest_float("z", -5.12, 5.12)
    return x**2 + y**2


def objective_dynamic_3(trial: optuna.Trial) -> float:
    """Dynamic search space function with varying parameters at specific trial numbers.

    Args:
        trial:
            The trial object to suggest parameters.

    Returns:
        Computed value of the function.
    """
    x = trial.suggest_float("x", -5.12, 5.12)
    y = trial.suggest_float("y", -5.12, 5.12)
    if trial.number == 0:
        z = trial.suggest_float("z", -5.12, 5.12)
    if 100 <= trial.number < 200:
        z = trial.suggest_float("z", -5.12, 5.12)
    if trial.number == 300:
        z = trial.suggest_float("z", -5.12, 5.12)
    return x**2 + y**2


# ruff: enable

# Mapping of objective functions
objective_map = {
    "Ackley": objective_Ackley,
    "sphere": objective_sphere,
    "Rastrigin": objective_Rastrigin,
    "Schwefel": objective_Schwefel,
    "ML": objective_ML,
    "dynamic_1": objective_dynamic_1,
    "dynamic_2": objective_dynamic_2,
    "dynamic_3": objective_dynamic_3,
}

# ---------------Settings---------------

# Toggle for running the benchmark
run_benchmark = False

# Choose a specific objective function for single experiment runs
objective_function_choice = "Rastrigin"
# Options: "Ackley", "sphere", "Rastrigin", "Schwefel", "ML", "dynamic_1", "dynamic_2", "dynamic_3"

# DE Sampler settings
population_size = "auto"
F = 0.8
CR = 0.9
debug = True

# Experiment configuration
num_experiments = 2  # Number of independent experiments
number_of_trials = 500  # Number of trials per experiment
n_jobs = 1
# for local loading
registry_root = "/home/j/PycharmProjects/optunahub-registry/package"
sampler = optunahub.load_local_module(
    package="samplers/llambo", registry_root=registry_root
).Sampler(debug=debug)

# ---------------Loading samplers---------------


# for remote loading
# sampler = optunahub.load_module("samplers/sampler").sampler(
#     debug=debug
# )

# Load the Random Sampler
sampler_rs = optuna.samplers.RandomSampler(seed=42)

# ---------------Experiments---------------

if run_benchmark:
    # Run the benchmark for multiple objective functions

    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # List of objective functions to evaluate
    objective_list = ["Ackley", "sphere", "Rastrigin", "Schwefel"]

    for objective_function_choice in objective_list:
        # Determine optimization direction
        direction = "maximize" if objective_function_choice == "Schwefel" else "minimize"
        minimize = direction == "minimize"

        # Get the mapped objective function
        objective_function = objective_map[objective_function_choice]

        # Initialize result storage
        results_de = np.zeros((num_experiments, number_of_trials))
        results_rs = np.zeros((num_experiments, number_of_trials))

        # Run experiments for the Sampler and Random Sampler
        for i in range(num_experiments):
            # Run DE Sampler
            study = optuna.create_study(sampler=sampler, direction=direction)
            study.optimize(objective_function, n_trials=number_of_trials, n_jobs=n_jobs)

            # Track Sampler's best values
            best_values_de = []
            current_best_de = float("inf") if minimize else float("-inf")
            for trial in study.trials:
                if trial.value is not None:
                    current_best_de = (
                        min(current_best_de, trial.value)
                        if minimize
                        else max(current_best_de, trial.value)
                    )
                    best_values_de.append(current_best_de)
            results_de[i, :] = best_values_de

            # Run Random Sampler
            study_rs = optuna.create_study(sampler=sampler_rs, direction=direction)
            study_rs.optimize(objective_function, n_trials=number_of_trials, n_jobs=n_jobs)

            # Track Random Sampler's best values
            best_values_rs = []
            current_best_rs = float("inf") if minimize else float("-inf")
            for trial in study_rs.trials:
                if trial.value is not None:
                    current_best_rs = (
                        min(current_best_rs, trial.value)
                        if minimize
                        else max(current_best_rs, trial.value)
                    )
                    best_values_rs.append(current_best_rs)
            results_rs[i, :] = best_values_rs

        # Compute and plot performance metrics
        mean_de = np.mean(results_de, axis=0)
        std_de = np.std(results_de, axis=0)
        mean_rs = np.mean(results_rs, axis=0)
        std_rs = np.std(results_rs, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(mean_de, label="Sampler (Mean Performance)", linestyle="-", color="blue")
        plt.fill_between(
            range(number_of_trials), mean_de - std_de, mean_de + std_de, color="blue", alpha=0.2
        )
        plt.plot(mean_rs, label="RandomSampler (Mean Performance)", linestyle="--", color="orange")
        plt.fill_between(
            range(number_of_trials), mean_rs - std_rs, mean_rs + std_rs, color="orange", alpha=0.2
        )
        plt.title(
            f"Performance Comparison ({objective_function_choice.capitalize()} - {direction.capitalize()})"
        )
        plt.xlabel("Trial Number")
        plt.ylabel("Objective Value (Log Scale)")
        plt.yscale("log")
        plt.grid(which="both", linestyle="--", linewidth=0.5)
        plt.legend()

        # Save plot to file
        filename = f"results/{objective_function_choice}_{direction}.png"
        plt.savefig(filename, dpi=300)
        plt.show()

else:
    # Run a single experiment for the chosen objective function

    direction = "maximize" if objective_function_choice == "Schwefel" else "minimize"
    minimize = direction == "minimize"

    # Get the mapped objective function
    objective_function = objective_map[objective_function_choice]

    results_de = np.zeros((num_experiments, number_of_trials))
    results_rs = np.zeros((num_experiments, number_of_trials))

    for i in range(num_experiments):
        # Run DE Sampler
        study = optuna.create_study(sampler=sampler, direction=direction)
        study.optimize(objective_function, n_trials=number_of_trials, n_jobs=n_jobs)

        best_values_de = []
        current_best_de = float("inf") if minimize else float("-inf")
        for trial in study.trials:
            if trial.value is not None:
                current_best_de = (
                    min(current_best_de, trial.value)
                    if minimize
                    else max(current_best_de, trial.value)
                )
                best_values_de.append(current_best_de)
        results_de[i, :] = best_values_de

        # Run Random Sampler
        study_rs = optuna.create_study(sampler=sampler_rs, direction=direction)
        study_rs.optimize(objective_function, n_trials=number_of_trials, n_jobs=n_jobs)

        best_values_rs = []
        current_best_rs = float("inf") if minimize else float("-inf")
        for trial in study_rs.trials:
            if trial.value is not None:
                current_best_rs = (
                    min(current_best_rs, trial.value)
                    if minimize
                    else max(current_best_rs, trial.value)
                )
                best_values_rs.append(current_best_rs)
        results_rs[i, :] = best_values_rs

    # Compute and display performance metrics
    mean_de = np.mean(results_de, axis=0)
    std_de = np.std(results_de, axis=0)
    mean_rs = np.mean(results_rs, axis=0)
    std_rs = np.std(results_rs, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_de, label="Sampler (Mean Performance)", linestyle="-", color="blue")
    plt.fill_between(
        range(number_of_trials), mean_de - std_de, mean_de + std_de, color="blue", alpha=0.2
    )
    plt.plot(mean_rs, label="RandomSampler (Mean Performance)", linestyle="--", color="orange")
    plt.fill_between(
        range(number_of_trials), mean_rs - std_rs, mean_rs + std_rs, color="orange", alpha=0.2
    )
    plt.title(
        f"Performance Comparison ({objective_function_choice.capitalize()} - {direction.capitalize()})"
    )
    plt.xlabel("Trial Number")
    plt.ylabel("Objective Value (Log Scale)")
    plt.yscale("log")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show()
