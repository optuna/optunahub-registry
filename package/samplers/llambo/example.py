import os
from typing import Any

import optuna
from optuna import Trial
import optunahub
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def objective_rf(trial: Trial) -> float:
    """Machine learning objective using RandomForestClassifier.
    Args:

        trial: The trial object to suggest hyperparameters.
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
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.01, step=0.001)

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
                    ccp_alpha=ccp_alpha,
                    random_state=42,
                ),
            ),
        ]
    )

    # Cross-validation for accuracy
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()


def get_llambo_sampler(use_azure: bool = False) -> Any:
    """Create and return a LLAMBO sampler based on configuration.

    Args:
        use_azure: Whether to use Azure OpenAI API.

    Returns:
        Configured LLAMBOSampler instance.
    """
    # sampler parameters
    params = {
        "custom_task_description": "Optimize RandomForest hyperparameters for digit classification.",
        "sm_mode": "discriminative",
        "max_requests_per_minute": 60,
        "n_initial_samples": 5,
    }

    if use_azure:
        # Verify required Azure environment variables
        required_vars = ["OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_API_VERSION"]
        missing_vars = [var for var in required_vars if var not in os.environ]

        if missing_vars:
            print(f"Missing required Azure environment variables: {missing_vars}")
            print("Falling back to standard OpenAI API...")
            return get_llambo_sampler(use_azure=False)

        print("Using Azure OpenAI API")
        return LLAMBOSampler(
            **params,
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o",
            azure=True,
            azure_api_base=os.environ["OPENAI_API_BASE"],
            azure_api_version=os.environ["OPENAI_API_VERSION"],
            azure_deployment_name="gpt-4o",
        )
    else:
        # Standard OpenAI API
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        print("Using standard OpenAI API")
        return LLAMBOSampler(
            **params,
            api_key=api_key,
            model="gpt-4o-mini",
        )


if __name__ == "__main__":
    # Configuration
    use_azure = True  # Set to True to use Azure OpenAI
    n_trials = 30
    n_jobs = 1

    # Load the LLAMBO sampler module
    module = optunahub.load_module("samplers/llambo")
    LLAMBOSampler = module.LLAMBOSampler

    # Create samplers
    llm_sampler = get_llambo_sampler(use_azure)
    random_sampler = optuna.samplers.RandomSampler(seed=42)

    # Create studies
    llm_study = optuna.create_study(sampler=llm_sampler, direction="maximize")
    random_study = optuna.create_study(sampler=random_sampler, direction="maximize")

    # Run optimization
    print("Running LLM-based optimization...")
    llm_study.optimize(objective_rf, n_trials=n_trials, n_jobs=n_jobs)

    print("Running random optimization...")
    random_study.optimize(objective_rf, n_trials=n_trials, n_jobs=n_jobs)

    # Print results
    print("\nLLM-based sampler results:")
    print(f"Best accuracy: {llm_study.best_value:.4f}")
    print(f"Best parameters: {llm_study.best_params}")

    print("\nRandom sampler results:")
    print(f"Best accuracy: {random_study.best_value:.4f}")
    print(f"Best parameters: {random_study.best_params}")
