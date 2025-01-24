import optuna
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def get_sample_hyperparameters(objective: callable) -> dict:
    """
    Generates a sample set of hyperparameters from the search space defined in the objective function.

    Args:
        objective: The Optuna objective function that defines the hyperparameter search space.

    Returns:
        A dictionary containing a sample of hyperparameters suggested by the objective's trial.
    """
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    try:
        objective(trial)
    except:
        pass  # Ignore errors post hyperparameter suggestion
    return trial.params


params = get_sample_hyperparameters(objective_ML)

print(params)
