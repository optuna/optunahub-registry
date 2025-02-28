from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optunahub
from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ---------------Objective Functions---------------


def objective_svm(trial: optuna.Trial) -> float:
    """Objective function for tuning SVM classifier."""
    data = load_digits()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter suggestions
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=C, kernel=kernel, gamma=gamma, random_state=42)),
        ]
    )

    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()


def objective_gb(trial: optuna.Trial) -> float:
    """Objective function for tuning Gradient Boosting classifier."""
    data = load_digits()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter suggestions
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 1.0)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    subsample = trial.suggest_uniform("subsample", 0.5, 1.0)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    subsample=subsample,
                    random_state=42,
                ),
            ),
        ]
    )

    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()


def objective_lr(trial: optuna.Trial) -> float:
    """Objective function for tuning Linear Regression."""
    data = load_digits()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter suggestions
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    # Note: normalize parameter is deprecated in sklearn 1.0 and removed in 1.2
    # Keeping it here for compatibility but not using it in the model
    _ = trial.suggest_categorical("normalize", [True, False])

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression(fit_intercept=fit_intercept)),
        ]
    )

    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scorer)
    return np.abs(scores.mean())


def objective_rf(trial: optuna.Trial) -> float:
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


# Mapping of objective functions
objective_map: Dict[str, Callable[[optuna.Trial], float]] = {
    "rf": objective_rf,
    "svm": objective_svm,
    "lr": objective_lr,
    "gb": objective_gb,
}

# ---------------Settings---------------

# Toggle for running the benchmark
run_benchmark = True

# Choose a specific objective function for single experiment runs
objective_function_choice = "rf"

# Sampler settings
sm_mode = "discriminative"
debug = True
model = "gpt-4o-mini"
max_requests_per_minute = 60
api_key = os.environ.get(
    "API_KEY",
    "",
)  # Set your API key here or via env variable

# Experiment configuration
num_experiments = 5  # Number of independent experiments
number_of_trials = 30  # Number of trials per experiment
n_jobs = 1

# For local loading
registry_root = "/home/j/PycharmProjects/optunahub-registry/package"


# ---------------Utility Functions for Saving/Loading Results---------------


def get_results_dir(objective_name: str, sampler_name: str) -> Path:
    """Create and return the directory path for saving results.

    Args:
        objective_name: Name of the objective function
        sampler_name: Name of the sampler ('llambo' or 'random')

    Returns:
        Path object for the directory
    """
    base_dir = Path("results")
    sampler_dir = base_dir / objective_name / sampler_name
    sampler_dir.mkdir(parents=True, exist_ok=True)
    return sampler_dir


def save_study(
    study: optuna.study.Study, objective_name: str, sampler_name: str, exp_num: int
) -> None:
    """Save study data to disk without pickling the actual study object.

    Args:
        study: The study object to save
        objective_name: Name of the objective function
        sampler_name: Name of the sampler
        exp_num: Experiment number
    """
    results_dir = get_results_dir(objective_name, sampler_name)

    # Extract trials data
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_dict = {
                "number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "datetime_start": str(trial.datetime_start),
                "datetime_complete": str(trial.datetime_complete),
            }
            trials_data.append(trial_dict)

    # Save trials data as JSON
    trials_path = results_dir / f"trials_exp{exp_num}.json"
    with open(trials_path, "w") as f:
        json.dump(trials_data, f)

    # Save best values as JSON for easier inspection
    best_values_path = results_dir / f"best_values_exp{exp_num}.json"

    # Extract the best values progression
    best_values = []
    direction_minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
    current_best = float("inf") if direction_minimize else float("-inf")

    for trial in study.trials:
        if trial.value is not None:
            current_best = (
                min(current_best, trial.value)
                if direction_minimize
                else max(current_best, trial.value)
            )
            best_values.append(current_best)

    # Save best values
    with open(best_values_path, "w") as f:
        json.dump(best_values, f)

    # Save study summary as text
    summary_path = results_dir / f"summary_exp{exp_num}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Best value: {study.best_value}\n")
        f.write(f"Best params: {study.best_params}\n")
        f.write(f"Direction: {study.direction}\n")


def load_study(objective_name: str, sampler_name: str, exp_num: int) -> Optional[bool]:
    """Check if study data exists on disk.

    Args:
        objective_name: Name of the objective function
        sampler_name: Name of the sampler
        exp_num: Experiment number

    Returns:
        True if study exists, None otherwise
    """
    results_dir = get_results_dir(objective_name, sampler_name)
    trials_path = results_dir / f"trials_exp{exp_num}.json"

    if trials_path.exists():
        # We don't actually load the study object, just check if it exists
        # and return a non-None value to indicate it exists
        return True
    return None


def load_best_values(
    objective_name: str, sampler_name: str, exp_num: int
) -> Optional[List[float]]:
    """Load best values from disk if they exist.

    Args:
        objective_name: Name of the objective function
        sampler_name: Name of the sampler
        exp_num: Experiment number

    Returns:
        List of best values or None if not found
    """
    results_dir = get_results_dir(objective_name, sampler_name)
    best_values_path = results_dir / f"best_values_exp{exp_num}.json"

    if best_values_path.exists():
        with open(best_values_path, "r") as f:
            return json.load(f)
    return None


def check_trial_progress(objective_name: str, sampler_name: str, exp_num: int) -> int:
    """Check how many trials have been completed for a specific experiment.

    Args:
        objective_name: Name of the objective function
        sampler_name: Name of the sampler ('llambo' or 'random')
        exp_num: Experiment number

    Returns:
        int: Number of completed trials, or 0 if no trials found
    """
    results_dir = get_results_dir(objective_name, sampler_name)
    trials_path = results_dir / f"trials_exp{exp_num}.json"

    if trials_path.exists():
        try:
            with open(trials_path, "r") as f:
                trials_data = json.load(f)
            return len(trials_data)
        except Exception as e:
            print(f"Error reading trials file: {e}")
            return 0
    return 0


def check_experiments_completed(objective_name: str, num_experiments: int) -> bool:
    """Check if all experiments for a given objective are already completed.

    Args:
        objective_name: Name of the objective function
        num_experiments: Number of experiments to check

    Returns:
        True if all experiments are completed, False otherwise
    """
    llambo_completed = True
    random_completed = True

    for exp_num in range(num_experiments):
        # Check if both LLAMBO and RandomSampler experiments are complete with enough trials
        llambo_trials = check_trial_progress(objective_name, "llambo", exp_num)
        random_trials = check_trial_progress(objective_name, "random", exp_num)

        if llambo_trials < number_of_trials:
            llambo_completed = False

        if random_trials < number_of_trials:
            random_completed = False

    return llambo_completed and random_completed


# Helper function to create distributions for loaded trials
def _suggest_distribution(
    param_name: str, param_value: Any
) -> optuna.distributions.BaseDistribution:
    """Create appropriate distribution objects for parameters based on their names and values.

    Args:
        param_name: Parameter name
        param_value: Parameter value

    Returns:
        Optuna distribution object
    """
    if param_name in ["C", "learning_rate"]:
        # These are likely log-uniform parameters
        return optuna.distributions.FloatDistribution(1e-5, 1e3, log=True)
    elif param_name in [
        "kernel",
        "gamma",
        "max_features",
        "bootstrap",
        "fit_intercept",
        "normalize",
    ]:
        # These are likely categorical parameters
        if param_name == "kernel":
            return optuna.distributions.CategoricalDistribution(
                ["linear", "poly", "rbf", "sigmoid"]
            )
        elif param_name == "gamma":
            return optuna.distributions.CategoricalDistribution(["scale", "auto"])
        elif param_name == "max_features":
            return optuna.distributions.CategoricalDistribution(["sqrt", "log2", None])
        elif param_name in ["bootstrap", "fit_intercept", "normalize"]:
            return optuna.distributions.CategoricalDistribution([True, False])
    elif param_name in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
        # These are likely integer parameters
        if param_name == "n_estimators":
            return optuna.distributions.IntDistribution(50, 300, log=False)
        elif param_name == "max_depth":
            return optuna.distributions.IntDistribution(3, 30, log=False)
        elif param_name == "min_samples_split":
            return optuna.distributions.IntDistribution(2, 15, log=False)
        elif param_name == "min_samples_leaf":
            return optuna.distributions.IntDistribution(1, 10, log=False)
    elif param_name == "subsample":
        return optuna.distributions.FloatDistribution(0.5, 1.0, log=False)

    # Default: return a reasonable uniform distribution
    return optuna.distributions.FloatDistribution(0.0, 1.0, log=False)


def resume_optimization(
    objective_function: Callable[[optuna.Trial], float],
    objective_name: str,
    sampler_name: str,
    exp_num: int,
    sampler: optuna.samplers.BaseSampler,
    direction: str,
    n_trials: int,
    n_jobs: int,
) -> Tuple[Optional[optuna.study.Study], Optional[List[float]]]:
    """Resume or start optimization with proper trial count.

    Args:
        objective_function: The objective function to optimize
        objective_name: Name of the objective function
        sampler_name: Name of the sampler ('llambo' or 'random')
        exp_num: Experiment number
        sampler: The sampler object to use
        direction: Optimization direction ('minimize' or 'maximize')
        n_trials: Total number of trials to run
        n_jobs: Number of parallel jobs

    Returns:
        study: The completed study or None if already completed
        best_values: List of best values throughout optimization or None
    """
    # Check existing progress
    results_dir = get_results_dir(objective_name, sampler_name)
    trials_path = results_dir / f"trials_exp{exp_num}.json"
    completed_trials = check_trial_progress(objective_name, sampler_name, exp_num)

    if completed_trials >= n_trials:
        print(
            f"{sampler_name.capitalize()} experiment {exp_num + 1} already completed with {completed_trials} trials"
        )
        best_values = load_best_values(objective_name, sampler_name, exp_num)
        return None, best_values

    # Create study with the proper direction
    study_direction = (
        optuna.study.StudyDirection.MINIMIZE
        if direction == "minimize"
        else optuna.study.StudyDirection.MAXIMIZE
    )
    study = optuna.create_study(sampler=sampler, direction=study_direction)

    # If we have previous trials, load them into the study
    if completed_trials > 0 and trials_path.exists():
        print(
            f"Resuming {sampler_name.capitalize()} experiment {exp_num + 1} from {completed_trials} trials"
        )

        # Load previous trials data
        with open(trials_path, "r") as f:
            trials_data = json.load(f)

        # Add previous trials to the study
        for trial_dict in trials_data:
            trial = optuna.trial.create_trial(
                params=trial_dict["params"],
                distributions={
                    param: _suggest_distribution(param, value)
                    for param, value in trial_dict["params"].items()
                },
                value=trial_dict["value"],
                state=optuna.trial.TrialState.COMPLETE,
            )
            study.add_trial(trial)

        remaining_trials = n_trials - completed_trials
    else:
        print(f"Starting new {sampler_name.capitalize()} experiment {exp_num + 1}")
        remaining_trials = n_trials

    # Set up callback for saving progress
    callback = SaveCallback(objective_name, sampler_name, exp_num)

    # Only run optimization if we have remaining trials
    if remaining_trials > 0:
        study.optimize(
            objective_function, n_trials=remaining_trials, n_jobs=n_jobs, callbacks=[callback]
        )

    # Extract best values
    best_values = load_best_values(objective_name, sampler_name, exp_num)
    if best_values is None:
        direction_minimize = direction == "minimize"
        best_values = []
        current_best = float("inf") if direction_minimize else float("-inf")

        for trial in study.trials:
            if trial.value is not None:
                current_best = (
                    min(current_best, trial.value)
                    if direction_minimize
                    else max(current_best, trial.value)
                )
                best_values.append(current_best)

    return study, best_values


# Callback to save study after each trial
class SaveCallback:
    def __init__(self, objective_name: str, sampler_name: str, exp_num: int) -> None:
        self.objective_name = objective_name
        self.sampler_name = sampler_name
        self.exp_num = exp_num

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        save_study(study, self.objective_name, self.sampler_name, self.exp_num)


# ---------------Loading samplers---------------


def create_samplers_for_objective(
    objective_name: str,
) -> Tuple[optuna.samplers.BaseSampler, optuna.samplers.BaseSampler]:
    """Create samplers specific to the objective function.

    Args:
        objective_name: Name of the objective function

    Returns:
        Tuple of (LLAMBO sampler, random sampler)
    """
    # Define a search space for LLAMBO sampler using the proper distribution types
    search_spaces = {
        "rf": {
            "n_estimators": optuna.distributions.IntDistribution(50, 300, log=False),
            "max_depth": optuna.distributions.IntDistribution(5, 30, log=False),
            "min_samples_split": optuna.distributions.IntDistribution(2, 15, log=False),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10, log=False),
            "max_features": optuna.distributions.CategoricalDistribution(["sqrt", "log2", None]),
            "bootstrap": optuna.distributions.CategoricalDistribution([True, False]),
        },
        "svm": {
            "C": optuna.distributions.FloatDistribution(1e-3, 1e3, log=True),
            "kernel": optuna.distributions.CategoricalDistribution(
                ["linear", "poly", "rbf", "sigmoid"]
            ),
            "gamma": optuna.distributions.CategoricalDistribution(["scale", "auto"]),
        },
        "lr": {
            "fit_intercept": optuna.distributions.CategoricalDistribution([True, False]),
            "normalize": optuna.distributions.CategoricalDistribution([True, False]),
        },
        "gb": {
            "n_estimators": optuna.distributions.IntDistribution(50, 300, log=False),
            "learning_rate": optuna.distributions.FloatDistribution(0.01, 1.0, log=True),
            "max_depth": optuna.distributions.IntDistribution(3, 30, log=False),
            "min_samples_split": optuna.distributions.IntDistribution(2, 15, log=False),
            "min_samples_leaf": optuna.distributions.IntDistribution(1, 10, log=False),
            "subsample": optuna.distributions.FloatDistribution(0.5, 1.0, log=False),
        },
    }

    # Create the LLAMBO sampler with the appropriate search space
    llambo_sampler = optunahub.load_module(
        package="samplers/llambo",
    ).LLAMBOSampler(
        api_key=api_key,
        model=model,
        debug=debug,
        sm_mode=sm_mode,
        max_requests_per_minute=max_requests_per_minute,
        search_space=search_spaces.get(objective_name),  # Using search_space parameter
        n_trials=number_of_trials,  # Add this line
    )

    # Create the Random sampler
    random_sampler = optuna.samplers.RandomSampler(seed=42)

    return llambo_sampler, random_sampler


# ---------------Experiments---------------


def run_experiments() -> None:
    """Run the benchmark experiments for all objective functions."""
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    # List of objective functions to evaluate
    objective_list = ["rf", "svm", "lr", "gb"]

    for objective_name in objective_list:
        print(f"Running experiments for {objective_name}...")

        # Check if all experiments for this objective are already completed
        if check_experiments_completed(objective_name, num_experiments):
            print(
                f"All experiments for {objective_name} are already completed. Skipping to visualization."
            )
            visualize_results(objective_name)
            continue

        # Determine optimization direction
        direction = "minimize" if objective_name == "lr" else "maximize"
        minimize = direction == "minimize"

        # Get the mapped objective function
        objective_function = objective_map[objective_name]

        # Create samplers specific to this objective function
        llambo_sampler, random_sampler = create_samplers_for_objective(objective_name)

        # Initialize result storage
        results_llambo = np.zeros((num_experiments, number_of_trials))
        results_random = np.zeros((num_experiments, number_of_trials))

        # Run experiments for the LLAMBO Sampler and Random Sampler
        for exp_num in range(num_experiments):
            print(f"Running experiment {exp_num + 1}/{num_experiments} for {objective_name}...")

            # Resume or start LLAMBO optimization
            _, best_values_llambo = resume_optimization(
                objective_function=objective_function,
                objective_name=objective_name,
                sampler_name="llambo",
                exp_num=exp_num,
                sampler=llambo_sampler,
                direction=direction,
                n_trials=number_of_trials,
                n_jobs=n_jobs,
            )

            # Ensure best_values_llambo has the right length
            if best_values_llambo and len(best_values_llambo) < number_of_trials:
                # Pad with the last best value
                last_best = (
                    best_values_llambo[-1]
                    if best_values_llambo
                    else (float("inf") if minimize else float("-inf"))
                )
                best_values_llambo.extend(
                    [last_best] * (number_of_trials - len(best_values_llambo))
                )

            if best_values_llambo:
                results_llambo[exp_num, :] = best_values_llambo[:number_of_trials]

            # Resume or start Random optimization
            _, best_values_random = resume_optimization(
                objective_function=objective_function,
                objective_name=objective_name,
                sampler_name="random",
                exp_num=exp_num,
                sampler=random_sampler,
                direction=direction,
                n_trials=number_of_trials,
                n_jobs=n_jobs,
            )

            # Ensure best_values_random has the right length
            if best_values_random and len(best_values_random) < number_of_trials:
                # Pad with the last best value
                last_best = (
                    best_values_random[-1]
                    if best_values_random
                    else (float("inf") if minimize else float("-inf"))
                )
                best_values_random.extend(
                    [last_best] * (number_of_trials - len(best_values_random))
                )

            if best_values_random:
                results_random[exp_num, :] = best_values_random[:number_of_trials]

        # Save the aggregated results
        save_aggregated_results(objective_name, results_llambo, results_random)

        # Visualize the results
        visualize_results(objective_name)


def save_aggregated_results(
    objective_name: str, results_llambo: np.ndarray, results_random: np.ndarray
) -> None:
    """Save aggregated results for all experiments of an objective.

    Args:
        objective_name: Name of the objective function
        results_llambo: Results from LLAMBO sampler
        results_random: Results from random sampler
    """
    base_dir = Path("results") / objective_name
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    np.save(base_dir / "results_llambo.npy", results_llambo)
    np.save(base_dir / "results_random.npy", results_random)

    # Also save as JSON for easier inspection
    results_dict = {"llambo": results_llambo.tolist(), "random": results_random.tolist()}

    with open(base_dir / "aggregated_results.json", "w") as f:
        json.dump(results_dict, f)


def load_aggregated_results(
    objective_name: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load aggregated results for all experiments of an objective.

    Args:
        objective_name: Name of the objective function

    Returns:
        Tuple of (LLAMBO results, random results) as numpy arrays
    """
    base_dir = Path("results") / objective_name

    # Try loading from numpy files first
    try:
        results_llambo = np.load(base_dir / "results_llambo.npy")
        results_random = np.load(base_dir / "results_random.npy")
        return results_llambo, results_random
    except Exception:
        # Fall back to JSON if numpy files don't exist
        try:
            with open(base_dir / "aggregated_results.json", "r") as f:
                results_dict = json.load(f)
            return np.array(results_dict["llambo"]), np.array(results_dict["random"])
        except Exception:
            # If neither exists, return None
            return None, None


def run_single_experiment(objective_name: str) -> None:
    """Run a single experiment for the chosen objective function.

    Args:
        objective_name: Name of the objective function
    """
    # Create samplers specific to this objective
    llambo_sampler, random_sampler = create_samplers_for_objective(objective_name)

    direction = "minimize" if objective_name == "lr" else "maximize"
    minimize = direction == "minimize"

    # Get the mapped objective function
    objective_function = objective_map[objective_name]

    # Check if experiments are already completed
    if check_experiments_completed(objective_name, num_experiments):
        print(
            f"All experiments for {objective_name} are already completed. Skipping to visualization."
        )
        visualize_results(objective_name)
        return

    results_llambo = np.zeros((num_experiments, number_of_trials))
    results_random = np.zeros((num_experiments, number_of_trials))

    for exp_num in range(num_experiments):
        print(f"Running experiment {exp_num + 1}/{num_experiments} for {objective_name}...")

        # Resume or start LLAMBO optimization
        _, best_values_llambo = resume_optimization(
            objective_function=objective_function,
            objective_name=objective_name,
            sampler_name="llambo",
            exp_num=exp_num,
            sampler=llambo_sampler,
            direction=direction,
            n_trials=number_of_trials,
            n_jobs=n_jobs,
        )

        # Ensure best_values_llambo has the right length
        if best_values_llambo and len(best_values_llambo) < number_of_trials:
            # Pad with the last best value
            last_best = (
                best_values_llambo[-1]
                if best_values_llambo
                else (float("inf") if minimize else float("-inf"))
            )
            best_values_llambo.extend([last_best] * (number_of_trials - len(best_values_llambo)))

        if best_values_llambo:
            results_llambo[exp_num, :] = best_values_llambo[:number_of_trials]

        # Resume or start Random optimization
        _, best_values_random = resume_optimization(
            objective_function=objective_function,
            objective_name=objective_name,
            sampler_name="random",
            exp_num=exp_num,
            sampler=random_sampler,
            direction=direction,
            n_trials=number_of_trials,
            n_jobs=n_jobs,
        )

        # Ensure best_values_random has the right length
        if best_values_random and len(best_values_random) < number_of_trials:
            # Pad with the last best value
            last_best = (
                best_values_random[-1]
                if best_values_random
                else (float("inf") if minimize else float("-inf"))
            )
            best_values_random.extend([last_best] * (number_of_trials - len(best_values_random)))

        if best_values_random:
            results_random[exp_num, :] = best_values_random[:number_of_trials]

    # Save aggregated results
    save_aggregated_results(objective_name, results_llambo, results_random)

    # Visualize results
    visualize_results(objective_name)


def visualize_results(objective_name: str) -> None:
    """Visualize the results for a specific objective.

    Args:
        objective_name: Name of the objective function
    """
    # Load the aggregated results
    results_llambo, results_random = load_aggregated_results(objective_name)

    if results_llambo is None or results_random is None:
        print(f"No results found for {objective_name}. Skipping visualization.")
        return

    # Determine optimization direction
    direction = "minimize" if objective_name == "lr" else "maximize"

    # Compute performance metrics
    mean_llambo = np.mean(results_llambo, axis=0)
    std_llambo = np.std(results_llambo, axis=0)
    mean_random = np.mean(results_random, axis=0)
    std_random = np.std(results_random, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_llambo, label="LLAMBO (Mean Performance)", linestyle="-", color="blue")
    plt.fill_between(
        range(len(mean_llambo)),
        mean_llambo - std_llambo,
        mean_llambo + std_llambo,
        color="blue",
        alpha=0.2,
    )
    plt.plot(mean_random, label="RandomSampler (Mean Performance)", linestyle="--", color="orange")
    plt.fill_between(
        range(len(mean_random)),
        mean_random - std_random,
        mean_random + std_random,
        color="orange",
        alpha=0.2,
    )
    plt.title(f"Performance Comparison ({objective_name.capitalize()} - {direction.capitalize()})")
    plt.xlabel("Trial Number")
    plt.ylabel("Objective Value (Log Scale)")
    plt.yscale("log")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    # Save plot to file
    filename = f"results/{objective_name}_{direction}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Visualization saved to {filename}")


# ---------------Main Execution---------------

if __name__ == "__main__":
    if run_benchmark:
        run_experiments()
    else:
        run_single_experiment(objective_function_choice)
