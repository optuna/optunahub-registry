"""SVM hyperparameter optimization benchmark for OptunaHub.

This module defines a cheap, real-world benchmark problem: tuning the
hyperparameters of a Support Vector Machine classifier (``sklearn.svm.SVC``)
on small built-in scikit-learn datasets. The objective is the k-fold
cross-validated classification error, which should be minimized.

The search space mixes log-scale continuous parameters (``C`` and ``gamma``),
a categorical parameter (``kernel``), and a conditional integer parameter
(``degree``, which only exists when ``kernel == "poly"``). This makes the
problem useful for checking how samplers handle mixed and conditional spaces.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import optuna
import optunahub


try:
    from sklearn.datasets import load_breast_cancer
    from sklearn.datasets import load_digits
    from sklearn.datasets import load_wine
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ModuleNotFoundError:
    raise ModuleNotFoundError("Please run `pip install scikit-learn` to use `svm_hpo`.")


# Built-in datasets that ship with scikit-learn, so no download is needed.
_DATASET_LOADERS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
    "digits": load_digits,
}

_KERNELS = ["linear", "rbf", "poly", "sigmoid"]

# Kernels for which the `gamma` parameter affects the model.
_GAMMA_KERNELS = {"rbf", "poly", "sigmoid"}

# Upper bound on solver iterations so that a bad configuration cannot hang a trial.
_MAX_ITER = 100_000


class Problem(optunahub.benchmarks.BaseProblem):
    """Tune an RBF/linear/polynomial/sigmoid SVM classifier with cross-validation.

    Args:
        dataset_name: Name of the built-in dataset. One of
            ``Problem.available_dataset_names``.
        n_splits: Number of folds for stratified k-fold cross-validation.
        seed: Seed that controls how the folds are shuffled.
    """

    available_dataset_names: list[str] = list(_DATASET_LOADERS.keys())

    def __init__(
        self,
        dataset_name: str = "breast_cancer",
        n_splits: int = 5,
        seed: int | None = None,
    ) -> None:
        if dataset_name not in _DATASET_LOADERS:
            raise ValueError(
                f"dataset_name must be one of {self.available_dataset_names}, "
                f"but got {dataset_name!r}."
            )
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, but got {n_splits}.")

        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self._X, self._y = _DATASET_LOADERS[dataset_name](return_X_y=True)
        self.reseed(seed)

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the full search space.

        Note that ``degree`` is only sampled when ``kernel == "poly"``.
        """
        return {
            "C": optuna.distributions.FloatDistribution(1e-3, 1e3, log=True),
            "gamma": optuna.distributions.FloatDistribution(1e-4, 1e1, log=True),
            "kernel": optuna.distributions.CategoricalDistribution(_KERNELS),
            "degree": optuna.distributions.IntDistribution(2, 5),
        }

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """The cross-validated error is minimized."""
        return [optuna.study.StudyDirection.MINIMIZE]

    def __call__(self, trial: optuna.Trial) -> float:
        """Sample a configuration from ``trial`` and evaluate it.

        This overrides the default behavior of ``BaseProblem`` so that
        ``degree`` is a conditional parameter: it is suggested only for the
        polynomial kernel, which is how it would be written in a real
        Optuna objective.
        """
        space = self.search_space
        params: dict[str, Any] = {
            "C": trial._suggest("C", space["C"]),
            "gamma": trial._suggest("gamma", space["gamma"]),
            "kernel": trial._suggest("kernel", space["kernel"]),
        }
        if params["kernel"] == "poly":
            params["degree"] = trial._suggest("degree", space["degree"])
        return self.evaluate(params)

    def evaluate(self, params: dict[str, Any]) -> float:
        """Return the k-fold cross-validated error (1 - accuracy) of the SVM.

        Args:
            params: A dictionary with ``C``, ``gamma`` and ``kernel``.
                ``degree`` is optional and only used by the polynomial kernel
                (defaults to 3, the scikit-learn default).

        Returns:
            The mean cross-validated classification error in ``[0, 1]``.
        """
        kernel = params["kernel"]
        if kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {_KERNELS}, but got {kernel!r}.")

        svc = SVC(
            C=float(params["C"]),
            kernel=kernel,
            gamma=float(params["gamma"]) if kernel in _GAMMA_KERNELS else "scale",
            degree=int(params.get("degree", 3)),
            max_iter=_MAX_ITER,
        )
        # SVMs are sensitive to feature scale, so standardize inside the pipeline
        # to avoid leaking statistics from the validation folds.
        model = make_pipeline(StandardScaler(), svc)
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self._cv_seed)

        with warnings.catch_warnings():
            # Hitting `max_iter` is expected for some extreme configurations.
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            scores = cross_val_score(model, self._X, self._y, cv=cv, scoring="accuracy")

        return float(1.0 - np.mean(scores))

    def reseed(self, seed: int | None = None) -> None:
        """Change the seed that controls how the cross-validation folds are shuffled."""
        self._cv_seed = int(np.random.RandomState(seed).randint(0, 2**31 - 1))
