from __future__ import annotations

import functools
import threading
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Literal
from typing import TypedDict
from typing import TypeVar

import numpy as np
import pandas as pd


# Define generic types for functions
F = TypeVar("F", bound=Callable[..., Any])


class RateLimiter:
    """
    A rate limiter that enforces a maximum number of requests per minute.

    This module is a complete different module from that of the original implementation.

    This implementation uses a simple time-based approach that enforces
    a consistent interval between requests.

    Attributes:
        min_interval (float): Minimum time interval between requests in seconds.
        last_request_time (float): Timestamp of the last request.
        lock (threading.RLock): Lock for thread safety.
    """

    def __init__(self, max_requests_per_minute: int = 60):
        """
        Initialize the rate limiter.

        Args:
            max_requests_per_minute (int): Maximum number of requests allowed per minute.
        """
        self.min_interval = 60.0 / max_requests_per_minute
        # Initialize with a time far in the past to allow the first request immediately
        self.last_request_time = time.time() - self.min_interval
        self.lock = threading.RLock()
        print(
            f"Rate limiter initialized: {max_requests_per_minute} requests/minute "
            f"({self.min_interval:.2f}s between requests)"
        )

    def wait_if_needed(self) -> None:
        """
        Wait if necessary to maintain the rate limit.
        """
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time

            # Calculate how long to wait to maintain the rate limit
            wait_time = max(0, self.min_interval - elapsed)

            if wait_time > 0:
                print(f"Rate limit: Waiting {wait_time:.2f}s before next request")
                time.sleep(wait_time)

            # Update the last request time after any required wait
            self.last_request_time = time.time()


def rate_limited(max_requests_per_minute: int = 60) -> Callable[[F], F]:
    """
    Decorator factory to rate limit a function.

    Args:
        max_requests_per_minute (int): Maximum number of requests allowed per minute.

    Returns:
        Callable: A decorator that rate limits the decorated function.
    """
    limiter = RateLimiter(max_requests_per_minute)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            limiter.wait_if_needed()
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


class OpenAIRateLimiter:
    """
    Rate limiter for OpenAI API calls.

    This class modifies an existing OpenAI interface to enforce
    rate limits on API calls.
    """

    def __init__(self, openai_interface: Any, max_requests_per_minute: int = 60):
        """
        Apply rate limiting to an OpenAI interface.

        Args:
            openai_interface: OpenAI interface to rate limit.
            max_requests_per_minute (int): Maximum requests allowed per minute.
        """
        self.limiter = RateLimiter(max_requests_per_minute)
        self.openai_interface = openai_interface

        # Store the original ask method
        self.original_ask = openai_interface.ask

        # Define a rate-limited version of the ask method
        @functools.wraps(self.original_ask)
        def rate_limited_ask(*args: Any, **kwargs: Any) -> Any:
            self.limiter.wait_if_needed()
            return self.original_ask(*args, **kwargs)

        # Replace the original ask method with our rate-limited version
        openai_interface.ask = rate_limited_ask

    def restore_original(self) -> None:
        """Restore the original ask method."""
        self.openai_interface.ask = self.original_ask


def apply_rate_limit(openai_instance: Any, max_requests_per_minute: int = 60) -> None:
    """
    Apply rate limiting to an OpenAI instance.

    This function modifies the provided OpenAI instance by wrapping its `ask` method
    with rate limiting functionality.

    Args:
        openai_instance: The OpenAI instance to rate limit.
        max_requests_per_minute (int): Maximum requests allowed per minute.
    """
    OpenAIRateLimiter(openai_instance, max_requests_per_minute)


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
