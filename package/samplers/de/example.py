from __future__ import annotations

import optuna
import optunahub
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np

import math

# ---------------Objective functions---------------
def objective_Ackley(trial: optuna.Trial) -> float :
    # Define the dimensionality of the problem
    n_dimensions = 10  # High-dimensional problem with 10 dimensions

    # Suggest a value for each dimension
    variables = [trial.suggest_float(f"x{i}" , -32.768 , 32.768) for i in range(n_dimensions)]

    # Compute the Ackley function value
    a = 20
    b = 0.2
    c = 2 * 3.141592653589793

    # Summation terms for the function
    sum_sq_term = sum(x ** 2 for x in variables)
    cos_term = sum(math.cos(c * x) for x in variables)

    # Ackley function formula
    result = -a * math.exp(-b * math.sqrt(sum_sq_term / n_dimensions)) - \
             math.exp(cos_term / n_dimensions) + a + math.exp(1)

    return result

def objective_sphere(trial: optuna.Trial) -> float:
    # Define the dimensionality of the problem
    n_dimensions = 10  # 10 dimensions

    # Suggest a value for each dimension
    variables = [trial.suggest_float(f"x{i}", -10.0, 10.0) for i in range(n_dimensions)]

    # Compute the summation of squares
    result = sum(x ** 2 for x in variables)

    return result


def objective_Rastrigin(trial: optuna.Trial) -> float:
    # Define the dimensionality of the problem
    n_dimensions = 10  # High-dimensional problem with 10 dimensions

    # Suggest a value for each dimension
    variables = [trial.suggest_float(f"x{i}", -5.12, 5.12) for i in range(n_dimensions)]

    # Rastrigin function parameters
    A = 10

    # Compute the Rastrigin function value
    sum_term = sum(x ** 2 - A * math.cos(2 * math.pi * x) for x in variables)

    # Rastrigin function formula
    result = A * n_dimensions + sum_term

    return result


def objective_Schwefel(trial: optuna.Trial) -> float:
    # Define the dimensionality of the problem
    n_dimensions = 10  # Example: High-dimensional problem with 10 dimensions

    # Suggest a value for each dimension
    variables = [trial.suggest_float(f"x{i}", -500, 500) for i in range(n_dimensions)]

    # Compute the Schwefel function value
    sum_term = sum(x * math.sin(math.sqrt(abs(x))) for x in variables)

    # Schwefel function formula
    result = 418.9829 * n_dimensions - sum_term

    # Return the negative for maximization purposes
    # Additionally added 10000 to make the result positive
    return - result + 10000



def objective_ML(trial: optuna.Trial) -> float :
    # Load dataset
    data = load_digits()
    X , y = data.data , data.target

    # Split into train and test sets
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.3 , random_state=42)

    # Define hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators" , 50 , 300)
    max_depth = trial.suggest_int("max_depth" , 5 , 30)
    min_samples_split = trial.suggest_int("min_samples_split" , 2 , 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf" , 1 , 10)
    max_features = trial.suggest_categorical("max_features" , ["sqrt" , "log2" , None])
    bootstrap = trial.suggest_categorical("bootstrap" , [True , False])

    # Build pipeline with scaling and RandomForestClassifier
    pipeline = Pipeline([
        ("scaler" , StandardScaler()) ,
        ("classifier" , RandomForestClassifier(
            n_estimators=n_estimators ,
            max_depth=max_depth ,
            min_samples_split=min_samples_split ,
            min_samples_leaf=min_samples_leaf ,
            max_features=max_features ,
            bootstrap=bootstrap ,
            random_state=42
            ))
        ])

    # Evaluate using cross-validation
    scores = cross_val_score(pipeline , X_train , y_train , cv=5 , scoring="accuracy")
    mean_accuracy = scores.mean()

    return mean_accuracy


def objective_dynamic_1(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x" , -5.12 , 5.12)
    y = trial.suggest_float("y" , -5.12 , 5.12)

    if trial.number == 0:
        z = trial.suggest_float("z", -5.12, 5.12)

    return x ** 2 + y ** 2

def objective_dynamic_2(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x" , -5.12 , 5.12)
    y = trial.suggest_float("y" , -5.12 , 5.12)
    if trial.number == 100 :
        z = trial.suggest_float("z" , -5.12 , 5.12)
    return x ** 2 + y ** 2


objective_map = {
    "Ackley": objective_Ackley,
    "sphere": objective_sphere,
    "Rastrigin": objective_Rastrigin,
    "Schwefel": objective_Schwefel,
    "ML": objective_ML,
    "dynamic_1": objective_dynamic_1,
    "dynamic_2": objective_dynamic_2,
}

# ---------------Settings---------------

# Paths
package_name = "package/samplers/de"
registry_root = "/home/j/experiments/optunahub-registry"

# Settings for the DE Sampler
population_size=50
F=0.8
CR=0.9
debug=True

# Local variable to set the direction: True for minimize, False for maximize
minimize = True

# Define the number of experiments and trials
num_experiments = 3
number_of_trials = 250

# Select the objective function
objective_function_choice = "dynamic_2"  # Choose objective function: "Ackley", "sphere", "Rastrigin", "Schwefel", "ML", "dynamic_1", "dynamic_2"


# ---------------Experiments and plotting---------------

# Map the objective function
objective_function = objective_map[objective_function_choice]

# Load the DE Sampler
sampler = optunahub.load_local_module(
    package=package_name ,
    registry_root=registry_root ,  # Path to the root of the optunahub-registry.
    ).DESampler(population_size=population_size , F=F , CR=CR, debug=debug)

# Load the Random Sampler
sampler_rs = optuna.samplers.RandomSampler(seed=42)  # Optional seed for reproducibility

# Store results for each experiment
results_de = np.zeros((num_experiments, number_of_trials))
results_rs = np.zeros((num_experiments, number_of_trials))

# Determine the optimization direction
direction = "minimize" if minimize else "maximize"

# Run multiple experiments
for i in range(num_experiments):
    # DE Sampler
    study = optuna.create_study(sampler=sampler, direction=direction)
    study.optimize(objective_function, n_trials=number_of_trials)

    # Track best values for DE Sampler
    best_values_de = []
    current_best_de = float("inf") if minimize else float("-inf")
    for trial in study.trials:
        if trial.value is not None:
            if minimize:
                current_best_de = min(current_best_de, trial.value)
            else:
                current_best_de = max(current_best_de, trial.value)
            best_values_de.append(current_best_de)
    results_de[i, :] = best_values_de

    # Random Sampler
    study_rs = optuna.create_study(sampler=sampler_rs, direction=direction)
    study_rs.optimize(objective_function, n_trials=number_of_trials)

    # Track best values for Random Sampler
    best_values_rs = []
    current_best_rs = float("inf") if minimize else float("-inf")
    for trial in study_rs.trials:
        if trial.value is not None:
            if minimize:
                current_best_rs = min(current_best_rs, trial.value)
            else:
                current_best_rs = max(current_best_rs, trial.value)
            best_values_rs.append(current_best_rs)
    results_rs[i, :] = best_values_rs

# Compute mean and standard deviation
mean_de = np.mean(results_de, axis=0)
std_de = np.std(results_de, axis=0)
mean_rs = np.mean(results_rs, axis=0)
std_rs = np.std(results_rs, axis=0)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(mean_de, linestyle='-', label='DESampler (Mean Performance)', color='blue')
plt.fill_between(range(number_of_trials), mean_de - std_de, mean_de + std_de, color='blue', alpha=0.2,
                 label='DESampler (Standard Deviation)')
plt.plot(mean_rs, linestyle='--', label='RandomSampler (Mean Performance)', color='orange')
plt.fill_between(range(number_of_trials), mean_rs - std_rs, mean_rs + std_rs, color='orange', alpha=0.2,
                 label='RandomSampler (Standard Deviation)')
plt.title(f'Comparison of DE Sampler and Random Sampler ({num_experiments} Experiments) - {objective_function_choice} - {direction.capitalize()}')
plt.xlabel('Trial Number')
plt.ylabel('Objective Value (Log Scale)')
plt.yscale('log')  # Always use log scale for the y-axis
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()