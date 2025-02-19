from __future__ import annotations

import time
from typing import Optional

import tiktoken


class RateLimiter:
    """
    A rate limiter that manages token and request limits over a specified time frame.

    This class implements a token bucket and request counter based rate limiting
    mechanism. It tracks both the number of tokens used and the number of requests
    made within a specified time frame, enforcing limits on both.

    Attributes:
        max_tokens (int): Maximum number of tokens allowed within the time frame.
        max_requests (int): Maximum number of requests allowed within the time frame.
        time_frame (float): Time window in seconds for which the limits apply.
        timestamps (list[float]): List of timestamps when requests were made.
        tokens_used (list[int]): List of token counts for each request.
        request_count (int): Current count of requests within the time frame.

    Example:
        >>> limiter = RateLimiter(max_tokens=4000, time_frame=60.0)
        >>> limiter.add_request(request_text="Hello, world!")
        >>> limiter.add_request(request_token_count=50)

    Notes:
        There is a known limitation where the rate limiter cannot anticipate if a
        request will exceed the limit before making it. This may lead to situations
        where limits are briefly exceeded before the limiter can respond.
    """

    def __init__(
        self,
        max_tokens: int,
        time_frame: float,
        max_requests: int = 700,
    ) -> None:
        """
        Initialize a new RateLimiter instance.

        Args:
            max_tokens (int): Maximum number of tokens allowed within time_frame.
            time_frame (float): Time window in seconds for which limits apply.
            max_requests (int, optional): Maximum number of requests allowed within
                time_frame. Defaults to 700.
        """
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.time_frame = time_frame
        self.timestamps: list[float] = []
        self.tokens_used: list[int] = []
        self.request_count = 0

    def add_request(
        self,
        request_text: Optional[str] = None,
        request_token_count: Optional[int] = None,
        current_time: Optional[float] = None,
    ) -> None:
        """
        Add a new request to the rate limiter and handle any necessary rate limiting.

        This method tracks the request and its token usage, removing old requests that
        fall outside the time frame. If either the token or request limit is exceeded,
        it will sleep for the appropriate duration.

        Args:
            request_text (Optional[str], optional): The text content of the request.
                Used to calculate token count. Defaults to None.
            request_token_count (Optional[int], optional): Pre-calculated token count
                for the request. Defaults to None.
            current_time (Optional[float], optional): Current timestamp for the request.
                Defaults to None, in which case the current time is used.

        Example:
            >>> limiter = RateLimiter(max_tokens=100, time_frame=60.0)
            >>> limiter.add_request(request_text="Test message")
            >>> limiter.add_request(request_token_count=10)

        Raises:
            ValueError: If neither request_text nor request_token_count is provided.
        """
        if current_time is None:
            current_time = time.time()

        # Remove old requests outside the time frame
        while self.timestamps and self.timestamps[0] < current_time - self.time_frame:
            self.timestamps.pop(0)
            self.tokens_used.pop(0)
            self.request_count -= 1

        # Add the new request timestamp
        self.timestamps.append(current_time)

        # Calculate token count from text or use provided count
        if request_text is not None:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            num_tokens = len(encoding.encode(request_text))
        elif request_token_count is not None:
            num_tokens = request_token_count
        else:
            raise ValueError("Either request_text or request_token_count must be specified.")

        self.tokens_used.append(num_tokens)
        self.request_count += 1

        # Handle request rate limiting
        if self.request_count >= self.max_requests:
            sleep_time = (self.timestamps[0] + self.time_frame) - current_time
            print(
                f"[Rate Limiter] Sleeping for {sleep_time:.2f}s to avoid hitting the request limit..."
            )
            time.sleep(sleep_time)
            self.request_count = 0

        # Handle token rate limiting
        if sum(self.tokens_used) > self.max_tokens:
            sleep_time = (self.timestamps[0] + self.time_frame) - current_time
            print(
                f"[Rate Limiter] Sleeping for {sleep_time:.2f}s to avoid hitting the token limit..."
            )
            time.sleep(sleep_time)
            self.timestamps.clear()
            self.tokens_used.clear()
