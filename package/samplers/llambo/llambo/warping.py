from __future__ import annotations

from typing import Literal
from typing import TypedDict

import numpy as np
import pandas as pd


class HyperparameterConstraint(TypedDict):
    """Type definition for hyperparameter constraints."""

    type: Literal["int", "float"]
    transform: Literal["log", "linear"]
    range: tuple[float, float]


class NumericalTransformer:
    """
    Transform numerical hyperparameters between original and warped search spaces.

    This class provides functionality to warp and unwarp numerical hyperparameters
    in a search space, particularly useful for optimization processes where certain
    parameters benefit from logarithmic scaling.

    Attributes:
        hyperparameter_constraints (dict[str, HyperparameterConstraint]): A dictionary
            mapping parameter names to their constraints and transformation rules.

    Example:
        >>> constraints = {
        ...     "learning_rate": {
        ...         "type": "float",
        ...         "transform": "log",
        ...         "range": (1e-4, 1e-1)
        ...     }
        ... }
        >>> transformer = NumericalTransformer(constraints)
        >>> config = pd.DataFrame({"learning_rate": [0.001]})
        >>> warped = transformer.warp(config)
        >>> warped["learning_rate"].iloc[0]
        -3.0
    """

    def __init__(self, hyperparameter_constraints: dict[str, HyperparameterConstraint]) -> None:
        """
        Initialize the NumericalTransformer with hyperparameter constraints.

        Args:
            hyperparameter_constraints: Dictionary mapping parameter names to their
                constraints including type, transformation method, and valid range.
        """
        self.hyperparameter_constraints = hyperparameter_constraints

    def warp(self, config: pd.DataFrame) -> pd.DataFrame:
        """
        Transform hyperparameters from their original space to the warped space.

        This method applies the specified transformations (e.g., logarithmic) to
        the hyperparameters according to their constraints.

        Args:
            config: DataFrame containing hyperparameter values to be transformed.

        Returns:
            pd.DataFrame: A new DataFrame with transformed hyperparameter values.

        Example:
            >>> constraints = {"param": {"type": "float", "transform": "log", "range": (0.1, 10)}}
            >>> transformer = NumericalTransformer(constraints)
            >>> config = pd.DataFrame({"param": [1.0]})
            >>> transformer.warp(config)
               param
            0   0.0

        Raises:
            AssertionError: If the number of columns in config doesn't match the
                number of hyperparameter constraints.
        """
        config_ = config.copy()
        assert len(config_.columns) == len(self.hyperparameter_constraints)

        for col in config_.columns:
            if col in self.hyperparameter_constraints:
                constraint = self.hyperparameter_constraints[col]
                param_type, transform = constraint["type"], constraint["transform"]

                if transform == "log":
                    assert param_type in ["int", "float"]
                    config_[col] = np.log10(config_[col])

        return config_

    def unwarp(self, config: pd.DataFrame) -> pd.DataFrame:
        """
        Transform hyperparameters from the warped space back to their original space.

        This method reverses the transformations applied by the warp method,
        converting values back to their original scale.

        Args:
            config: DataFrame containing warped hyperparameter values.

        Returns:
            pd.DataFrame: A new DataFrame with un-warped hyperparameter values.

        Example:
            >>> constraints = {"param": {"type": "float", "transform": "log", "range": (0.1, 10)}}
            >>> transformer = NumericalTransformer(constraints)
            >>> config = pd.DataFrame({"param": [0.0]})
            >>> transformer.unwarp(config)
               param
            0   1.0

        Raises:
            AssertionError: If the number of columns in config doesn't match the
                number of hyperparameter constraints.
        """
        config_ = config.copy()
        assert len(config_.columns) == len(self.hyperparameter_constraints)

        for col in config_.columns:
            if col in self.hyperparameter_constraints:
                constraint = self.hyperparameter_constraints[col]
                param_type, transform = constraint["type"], constraint["transform"]

                if transform == "log":
                    assert param_type in ["int", "float"]
                    config_[col] = 10 ** config_[col]

        return config_
