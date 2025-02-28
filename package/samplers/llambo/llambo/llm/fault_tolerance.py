from __future__ import annotations

import multiprocessing
from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeVar
import warnings


# Type variables for generic function types
T = TypeVar("T")
R = TypeVar("R")


def overtime_kill(
    target_function: Callable[..., Any],
    target_function_args: tuple[Any, ...] | None = None,
    time_limit: int = 60,
    ret: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """
    Run a target function with a time limit and terminate if it exceeds the limit.

    This function executes the target function in a separate process and monitors its execution
    time. If the function exceeds the specified time limit, it will be terminated.

    Args:
        target_function: The function to be executed with timeout monitoring.
        target_function_args: Optional tuple of arguments to pass to the target function.
        time_limit: Maximum execution time allowed in seconds.
        ret: Whether the target function returns data that needs to be captured.

    Returns:
        tuple[bool, dict[str, Any]]: A tuple containing:
            - bool: True if execution exceeded time limit, False otherwise.
            - dict: Captured return data from the target function.

    Example:
        >>> def long_task(ret_dict):
        ...     import time
        ...
        ...     time.sleep(2)
        ...     ret_dict["result"] = 42
        ...
        >>> exceeded, result = overtime_kill(long_task, time_limit=3)
        The operation finishes in time
        >>> print(exceeded, result)
        False {'result': 42}

    Notes:
        The target function should accept a dictionary as its first argument if ret=True,
        which will be used to store return values that need to be passed back to the
        main process.
    """
    ret_dict = multiprocessing.Manager().dict()

    if target_function_args is not None:
        p = multiprocessing.Process(
            target=target_function,
            args=(ret_dict,) + target_function_args,
        )
    elif ret:
        p = multiprocessing.Process(target=target_function, args=(ret_dict,))
    else:
        p = multiprocessing.Process(target=target_function)

    p.start()
    p.join(time_limit)

    if p.is_alive():
        warnings.warn(
            f"The operation takes longer than {time_limit} seconds, terminating the execution...",
            UserWarning,
            stacklevel=2,
        )
        p.terminate()
        p.join()
        return True, dict(ret_dict)

    print("The operation finishes in time")
    return False, dict(ret_dict)


def retry_overtime_kill(
    target_function: Callable[..., Any],
    target_function_args: tuple[Any, ...] | None = None,
    time_limit: int = 60,
    maximum_retry: int = 3,
    ret: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """
    Run a target function with retries if it exceeds the time limit.

    This function attempts to execute the target function multiple times if it
    exceeds the specified time limit, up to a maximum number of retries.

    Args:
        target_function: The function to be executed with timeout monitoring.
        target_function_args: Optional tuple of arguments to pass to the target function.
        time_limit: Maximum execution time allowed in seconds per attempt.
        maximum_retry: Maximum number of retry attempts if time limit is exceeded.
        ret: Whether the target function returns data that needs to be captured.

    Returns:
        tuple[bool, dict[str, Any]]: A tuple containing:
            - bool: True if all retries failed, False if execution succeeded.
            - dict: Captured return data from the target function.

    Example:
        >>> def unstable_task(ret_dict):
        ...     import time
        ...
        ...     time.sleep(2)
        ...     ret_dict["result"] = 42
        ...
        >>> exceeded, result = retry_overtime_kill(unstable_task, time_limit=3)
        Attempt 1 of 3
        The operation finishes in time
        >>> print(exceeded, result)
        False {'result': 42}
    """
    for attempt in range(maximum_retry):
        print(f"Operation under time limit: attempt {attempt + 1} of {maximum_retry}")
        exceeded, result = overtime_kill(target_function, target_function_args, time_limit, ret)

        if not exceeded:
            return False, result

        print("Retrying...")

    warnings.warn(
        "All retries exhausted. The operation failed to complete within the time limit.",
        UserWarning,
        stacklevel=2,
    )
    return True, {}


def retry_overtime_decorator(
    time_limit: int = 60,
    maximum_retry: int = 3,
    ret: bool = True,
) -> Callable[[Callable[..., R]], Callable[..., Optional[R]]]:
    """
    Create a decorator that adds timeout and retry functionality to a function.

    This decorator wraps a function to provide retry functionality when the function
    execution exceeds a specified time limit. It supports both regular functions
    and class methods, handling the self parameter appropriately. Compared to
    `retry_overtime_kill`, this decorator could be more efficient. However, this
    decorator could be less convenient in a class, where the class instance is not
    available for arguments of the decorator.

    Args:
        time_limit: Maximum execution time allowed in seconds per attempt.
        maximum_retry: Maximum number of retry attempts if time limit is exceeded.
        ret: Whether the function returns data that needs to be captured.

    Returns:
        Callable: A decorator function that adds timeout and retry functionality.

    Example:
        >>> @retry_overtime_decorator(time_limit=3, maximum_retry=2)
        ... def example_task(ret_dict):
        ...     import time
        ...
        ...     time.sleep(2)
        ...     ret_dict["result"] = 42
        ...
        >>> result = example_task()
        Attempt 1 of 2
        The operation finishes in time
        >>> print(result)
        42
    """

    def decorator(target_function: Callable[..., R]) -> Callable[..., Optional[R]]:
        def wrapper(*args: Any, **kwargs: Any) -> Optional[R]:
            # Handle class methods vs regular functions
            if args and hasattr(args[0], "__class__"):
                self_instance = args[0]
                message = args[1]
                target_function_args = ((message,), kwargs)

                def target_function_with_self(
                    ret_dict: dict[str, Any], *func_args: Any
                ) -> Any:  # Changed from -> None
                    return target_function(self_instance, func_args[0][0], ret_dict)

            else:
                target_function_args = (args, kwargs)

                def target_function_with_self(
                    ret_dict: dict[str, Any],
                    *func_args: Any,
                ) -> Any:  # Changed from -> None
                    return target_function(ret_dict, *func_args[0], **func_args[1])

            exceeded, result = retry_overtime_kill(
                target_function=target_function_with_self,
                target_function_args=target_function_args,
                time_limit=time_limit,
                maximum_retry=maximum_retry,
                ret=ret,
            )

            return result.get("result") if result else None

        return wrapper

    return decorator
