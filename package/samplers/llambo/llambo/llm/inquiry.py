from __future__ import annotations

import ast
import traceback
from typing import Any
from typing import Callable
from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat import ChatCompletionMessageParam

from llambo.llm.cost import Calculator
from llambo.llm.fault_tolerance import retry_overtime_kill


class LLMBase:
    """
    Base class for all LLMs.

    This class serves as the foundation for different language model implementations,
    providing common functionality and attributes.

    Attributes:
        api_key (Optional[str]): The API key for authentication.
        model (str): The LLM model identifier being used.
        debug (bool): Flag indicating if debug mode is enabled.

    Example:
        >>> base_llm = LLMBase(api_key="your-key", model="gpt-4", debug=True)
        >>> print(base_llm.model)
        'gpt-4'
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gpt-4-mini",
        timeout: float = 60,
        maximum_generation_attempts: int = 3,
        maximum_timeout_attempts: int = 3,
        debug: bool = False,
    ) -> None:
        """
        Initialize the base LLM.

        Args:
            api_key (Optional[str]): The API key for authentication.
            model (str, optional): The LLM model identifier to use. Defaults to 'gpt-4-mini'.
            debug (bool, optional): Enable debug mode for detailed logging. Defaults to False.
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.maximum_generation_attempts = maximum_generation_attempts
        self.maximum_timeout_attempts = maximum_timeout_attempts
        self.debug = debug


class OpenAI_interface(LLMBase):
    """
    A client for interacting with OpenAI's interface

    This class provides methods to communicate with models through OpenAI's API,
    with built-in retry functionality for handling timeouts.

    Attributes:
        timeout (int): Maximum time limit for API calls.
        maximum_retry (int): Maximum number of retry attempts.
        client (OpenAI): The OpenAI client instance for making API calls.

    Example:
        >>> gpt = OpenAI(api_key="your-key", model="gpt-4")
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = gpt.ask(messages)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-mini",
        timeout: float = 60,
        maximum_generation_attempts: int = 3,
        maximum_timeout_attempts: int = 3,
        debug: bool = False,
    ) -> None:
        """
        Initialize the OpenAI client.

        Args:
            api_key (str): The OpenAI API key for authentication.
            model (str, optional): The model identifier to use. Defaults to 'gpt-4-mini'.
            debug (bool, optional): Enable debug mode for detailed logging. Defaults to False.
        """
        super().__init__(
            api_key, model, timeout, maximum_generation_attempts, maximum_timeout_attempts, debug
        )

        if self.model == "deepseek-chat" or self.model == "deepseek-reasoner":
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            self.client = OpenAI(api_key=api_key)

    @staticmethod
    def print_prompt(messages: list[ChatCompletionMessageParam]) -> None:
        """
        Print each segment of a message prompt.

        Args:
            messages (list[ChatCompletionMessageParam]): List of message segments to print.

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Hello!"},
            ... ]
            >>> OpenAI_interface.print_prompt(messages)
        """
        for message in messages:
            if isinstance(message["content"], str):
                print(message["content"])

    def ask_base(
        self,
        messages: list[ChatCompletionMessageParam],
        ret_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[str], float]:
        """
        Base method to send a message to the chat model and capture the response.

        Args:
            messages (list[ChatCompletionMessageParam]): The messages to be sent to the chat model.
            ret_dict (Optional[dict[str, Any]], optional): A dictionary to capture the
                method's return value. Defaults to None.

        Returns:
            tuple[Optional[str], float]: The chat model's response text and the cost,
                or (None, 0.0) if the request fails.
        """
        if self.debug:
            print("---Prompt beginning marker---")
            self.print_prompt(messages)
            print("---Prompt ending marker---")

        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        if response.choices[0].message.content is None:
            return None, 0.0

        response_text: str = response.choices[0].message.content

        if self.debug:
            print("---Response beginning marker---")
            print(response_text)
            print("---Response ending marker---")

        calculator_instance = Calculator(self.model, messages, response_text)

        if self.model == "deepseek-chat" or self.model == "deepseek-reasoner":
            cost = calculator_instance.calculate_cost_DeepSeek()
        else:
            cost = calculator_instance.calculate_cost_GPT()

        if ret_dict is not None:
            ret_dict["result"] = (response_text, cost)

        return response_text, cost

    def ask(
        self,
        messages: list[ChatCompletionMessageParam],
        ret_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[str], float]:
        """
        Send a message to the chat model with retry functionality for handling timeouts.

        Args:
            messages (list[ChatCompletionMessageParam]): The messages to be sent to the chat model.
            ret_dict (Optional[dict[str, Any]], optional): A dictionary to capture the
                method's return value. Defaults to None.

        Returns:
            tuple[Optional[str], float]: The chat model's response text and cost,
                or ("termination_signal", cost) if the request times out.
        """

        def target_function(ret_dict: dict[str, Any], *args: Any) -> None:
            # We need to unpack args properly here - the first arg is messages
            messages = args[0]
            self.ask_base(messages, ret_dict=ret_dict)

        exceeded, result = retry_overtime_kill(
            target_function=target_function,
            target_function_args=(messages,),
            time_limit=self.timeout,
            maximum_retry=self.maximum_timeout_attempts,
            ret=True,
        )

        response_text, cost = result.get("result")

        if not exceeded:
            return response_text, cost
        else:
            return "termination_signal", cost

    def ask_with_test(
        self,
        messages: list[ChatCompletionMessageParam],
        tests: Callable[[str], Any],
    ) -> tuple[Any, float]:
        """
        This method is only for simple testing functions with retry, such as testing general
        strings or Python objects (instead of multiple lines of Python code).

        Tests are also supposed to convert the response to the expected type.

        Args:
            messages: The messages to be sent to the chat model.
            tests: A function to test the response from the chat model.

        Returns:
            tuple[Any, float]: The tested response and the accumulated cost.
        """
        cost_accumulation = 0.0

        def target_function(ret_dict: dict[str, Any], *args: Any) -> None:
            response, cost = self.ask_base(*args)
            ret_dict["response"] = response
            ret_dict["cost"] = cost

        for trial_count in range(self.maximum_generation_attempts):
            print(
                f"Sequence generation under testing: attempt {trial_count + 1} of {self.maximum_generation_attempts}"
            )
            exceeded, result = retry_overtime_kill(
                target_function=target_function,
                target_function_args=(messages,),
                time_limit=self.timeout,
                maximum_retry=self.maximum_timeout_attempts,
                # This retry is for timeout, instead of tests
                ret=True,
            )

            if exceeded:
                print(f"Inquiry timed out for {self.maximum_timeout_attempts} times, retrying...")
                continue

            response = result.get("response")
            cost = result.get("cost", 0.0)
            cost_accumulation += cost

            try:
                response = tests(response)
                print("Test passed")
                return response, cost_accumulation
            except Exception:
                print("Test failed, reason:")
                print(traceback.format_exc())
                print("Trying again")

        print("Maximum trial reached for sequence generation under testing")
        return "termination_signal", cost_accumulation


def extract_code_base(raw_sequence: str, language: str = "python") -> str:
    """
    Extract code from a raw text sequence based on code block markers.

    Args:
        raw_sequence (str): The raw text containing code blocks.
        language (str, optional): The programming language to extract. Defaults to "python".

    Returns:
        str: The extracted code or the original sequence if no code blocks are found.
    """
    # Try to find code block markers with language specification
    try:
        sub1 = f"```{language}"
        idx1 = raw_sequence.index(sub1)
    except ValueError:
        try:
            sub1 = f"``` {language}"
            idx1 = raw_sequence.index(sub1)
        except ValueError:
            try:
                sub1 = "```"
                idx1 = raw_sequence.index(sub1)
            except ValueError:
                return raw_sequence

    # Find the closing code block marker
    sub2 = "```"
    try:
        idx2 = raw_sequence.index(
            sub2,
            idx1 + 1,
        )
        extraction = raw_sequence[idx1 + len(sub1) + 1 : idx2]
        return extraction
    except ValueError:
        return raw_sequence


def extract_code(raw_sequence: str, language: str = "python", mode: str = "code") -> Any:
    """
    Extract and optionally parse code from a raw text sequence.

    Args:
        raw_sequence (str): The raw text containing code blocks.
        language (str, optional): The programming language to extract. Defaults to "python".
        mode (str, optional): The extraction mode, either "code" or "python_object". Defaults to "code".

    Returns:
        Any: The extracted code as a string or a Python object based on the mode.
    """
    extraction = extract_code_base(raw_sequence, language)
    if mode == "code":
        return extraction
    if mode == "python_object":
        return ast.literal_eval(extraction)
