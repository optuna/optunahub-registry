from __future__ import annotations

import functools
import threading
import time
from typing import Any, Callable

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
        print(f"Rate limiter initialized: {max_requests_per_minute} requests/minute "
              f"({self.min_interval:.2f}s between requests)")

    def wait_if_needed(self):
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


def rate_limited(max_requests_per_minute: int = 60):
    """
    Decorator factory to rate limit a function.

    Args:
        max_requests_per_minute (int): Maximum number of requests allowed per minute.

    Returns:
        Callable: A decorator that rate limits the decorated function.
    """
    limiter = RateLimiter(max_requests_per_minute)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait_if_needed()
            return func(*args, **kwargs)

        return wrapper

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
        def rate_limited_ask(*args, **kwargs):
            self.limiter.wait_if_needed()
            return self.original_ask(*args, **kwargs)

        # Replace the original ask method with our rate-limited version
        openai_interface.ask = rate_limited_ask

    def restore_original(self):
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