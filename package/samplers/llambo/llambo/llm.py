from __future__ import annotations

import multiprocessing
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar
import warnings

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import tiktoken
import transformers


class Calculator:
    """
    A helper class that estimates and calculates token usage/cost for GPT and DeepSeek models.

    Attributes:
        model (str): Name of the model (e.g., 'gpt-4o', 'deepseek-chat', etc.).
        formatted_input_sequence (Optional[List[Dict[str, str]]]): The input in the typical
            OpenAI chat-completion 'messages' format.
        output_sequence_string (Optional[str]): The raw string output from the model.
        input_token_length (int): Count of how many tokens appear in the input.
        output_token_length (int): Count of how many tokens appear in the output.
    """

    # Pricing per 1M input tokens in USD for GPT models
    GPT_input_pricing: Dict[str, float] = {
        "gpt-4o": 2.50,
        "gpt-4o-2024-11-20": 2.50,
        "gpt-4o-2024-08-06": 2.50,
        "gpt-4o-2024-05-13": 5.00,
        "gpt-4o-mini": 0.15,
        "gpt-4o-mini-2024-07-18": 0.15,
        "o1-preview": 15.00,
        "o1-preview-2024-09-12": 15.00,
        "o1-mini": 3.00,
        "o1-mini-2024-09-12": 3.00,
        "chatgpt-4o-latest": 5.00,
        "gpt-4-turbo": 10.00,
        "gpt-4-turbo-2024-04-09": 10.00,
        "gpt-4": 30.00,
        "gpt-4-32k": 60.00,
        "gpt-4-0125-preview": 10.00,
        "gpt-4-1106-preview": 10.00,
        "gpt-4-vision-preview": 10.00,
        "gpt-3.5-turbo-0125": 0.50,
        "gpt-3.5-turbo-instruct": 1.50,
        "gpt-3.5-turbo-1106": 1.00,
        "gpt-3.5-turbo-0613": 1.50,
        "gpt-3.5-turbo-16k-0613": 3.00,
        "gpt-3.5-turbo-0301": 1.50,
        "davinci-002": 2.00,
        "babbage-002": 0.40,
    }

    # Pricing per 1M output tokens in USD for GPT models
    GPT_output_pricing: Dict[str, float] = {
        "gpt-4o": 10.00,
        "gpt-4o-2024-11-20": 10.00,
        "gpt-4o-2024-08-06": 10.00,
        "gpt-4o-2024-05-13": 15.00,
        "gpt-4o-mini": 0.60,
        "gpt-4o-mini-2024-07-18": 0.60,
        "o1-preview": 60.00,
        "o1-preview-2024-09-12": 60.00,
        "o1-mini": 12.00,
        "o1-mini-2024-09-12": 12.00,
        "chatgpt-4o-latest": 15.00,
        "gpt-4-turbo": 30.00,
        "gpt-4-turbo-2024-04-09": 30.00,
        "gpt-4": 60.00,
        "gpt-4-32k": 120.00,
        "gpt-4-0125-preview": 30.00,
        "gpt-4-1106-preview": 30.00,
        "gpt-4-vision-preview": 30.00,
        "gpt-3.5-turbo-0125": 1.50,
        "gpt-3.5-turbo-instruct": 2.00,
        "gpt-3.5-turbo-1106": 2.00,
        "gpt-3.5-turbo-0613": 2.00,
        "gpt-3.5-turbo-16k-0613": 4.00,
        "gpt-3.5-turbo-0301": 2.00,
        "davinci-002": 2.00,
        "babbage-002": 0.40,
    }

    DeepSeek_input_pricing: Dict[str, float] = {
        "deepseek-chat": 0.14,
        "deepseek-reasoner": 0.55,
    }
    DeepSeek_output_pricing: Dict[str, float] = {
        "deepseek-chat": 0.28,
        "deepseek-reasoner": 2.19,
    }

    def __init__(
        self,
        model: str,
        formatted_input_sequence: Optional[List[Dict[str, str]]] = None,
        output_sequence_string: Optional[str] = None,
    ) -> None:
        """
        Args:
            model: The name of the model used (e.g. 'gpt-4o').
            formatted_input_sequence: The input (i.e. OpenAI-style messages) used for the query.
            output_sequence_string: The string output from the model.
        """
        self.model = model
        self.formatted_input_sequence = formatted_input_sequence
        self.output_sequence_string = output_sequence_string
        self.input_token_length: int = 0
        self.output_token_length: int = 0

    def calculate_input_token_length_GPT(self) -> int:
        """
        Calculate the number of tokens used in the input prompt (for GPT-like models)
        using the tiktoken library.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        if self.formatted_input_sequence:
            for message in self.formatted_input_sequence:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
        num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def calculate_output_token_length_GPT(self) -> int:
        """
        Calculate the number of tokens in the model's output (for GPT-like models).
        Uses tiktoken to encode the output string.
        """
        tokenizer = tiktoken.get_encoding("cl100k_base")
        if self.output_sequence_string:
            tokens = tokenizer.encode(self.output_sequence_string)
            return len(tokens)
        return 0

    def calculate_cost_GPT(self) -> float:
        """
        Calculate approximate cost for a GPT-based model using both the input and output lengths.
        """
        # Input token calculation
        if self.formatted_input_sequence is not None:
            self.input_token_length = self.calculate_input_token_length_GPT()
            input_cost = self.input_token_length * self.GPT_input_pricing[self.model] / 1e6
        else:
            input_cost = 0

        # Output token calculation
        if self.output_sequence_string is not None:
            self.output_token_length = self.calculate_output_token_length_GPT()
            output_cost = self.output_token_length * self.GPT_output_pricing[self.model] / 1e6
        else:
            output_cost = 0

        return input_cost + output_cost

    def calculate_token_length_DeepSeek(self) -> None:
        """
        Tokenization logic for DeepSeek-based models using a Hugging Face tokenizer.
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_relative_path = "tokenizers/deepseek"
        tokenizer_absolute_path = os.path.join(module_dir, tokenizer_relative_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_absolute_path, trust_remote_code=True
        )

        if self.formatted_input_sequence is not None:
            # Convert the OpenAI-like message format to a single prompt string
            input_sequence = PromptBase.formatted_to_string_OpenAI(self.formatted_input_sequence)
            input_tokenized = tokenizer.encode(input_sequence)
            self.input_token_length = len(input_tokenized)

        if self.output_sequence_string is not None:
            output_tokenized = tokenizer.encode(self.output_sequence_string)
            self.output_token_length = len(output_tokenized)

    def calculate_cost_DeepSeek(self) -> float:
        """
        Approximate cost for DeepSeek-based models. Must first calculate token lengths.
        """
        self.calculate_token_length_DeepSeek()
        cost = (
            self.input_token_length * self.DeepSeek_input_pricing[self.model]
            + self.output_token_length * self.DeepSeek_output_pricing[self.model]
        )
        cost /= 1e6
        return cost


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
        target_function (Callable[..., Any]): The function to execute.
        target_function_args (tuple[Any, ...] | None): Optional arguments to pass to the function.
        time_limit (int): Maximum execution time allowed in seconds.
        ret (bool): If True, captures return data from the target function using a Manager dict.

    Returns:
        (bool, dict[str, Any]):
            bool: True if the time limit was exceeded, else False.
            dict[str, Any]: Data captured from the target function if ret=True.
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

    Args:
        target_function (Callable[..., Any]): The function to execute.
        target_function_args (tuple[Any, ...] | None): Optional arguments to pass to the function.
        time_limit (int): Max execution time in seconds per attempt.
        maximum_retry (int): Maximum number of retries.
        ret (bool): If True, captures return data from the target function.

    Returns:
        (bool, dict[str, Any]):
            bool: True if all retries exceeded the time limit, else False.
            dict[str, Any]: Data from the target function if available.
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


class PromptBase:
    """
    A base class for managing and formatting prompts for Language Learning Models (LLMs).

    This class provides functionality to store, format, and display prompts in a format
    compatible with GPT-style language models. It maintains an internal list of formatted
    prompts and provides methods to manipulate and view them.
    """

    def __init__(self) -> None:
        """
        Initialize a new PromptBase instance with an empty prompt list and conversation history.
        """
        self.prompt: Any = None
        self.input_history: list[str] = []
        self.output_history: list[str] = []
        self.conversation_history: list[str] = []

    @staticmethod
    def list_to_formatted_OpenAI(prompt_as_list: list[str]) -> list[dict[str, str]]:
        """
        Format a list of prompt strings into a format compatible with OpenAI chat completions.

        Args:
            prompt_as_list (list[str]): A list of separate prompt strings.

        Returns:
            list[dict[str, str]]: A single-element list of dict with "role" and "content".
        """
        combined_content = "\n".join(prompt_as_list)
        formatted_prompt = [{"role": "user", "content": combined_content}]
        return formatted_prompt

    @staticmethod
    def formatted_to_string_OpenAI(formatted_prompt: list[dict[str, str]]) -> str:
        """
        Convert an OpenAI-style list[dict{role, content}] into a single string,
        ignoring the 'role' keys.

        Args:
            formatted_prompt (list[dict[str, str]]): The formatted messages.

        Returns:
            str: Combined content of all message segments.
        """
        concatenated_string = ""
        for segment in formatted_prompt:
            concatenated_string += segment["content"]
        return concatenated_string

    def print_prompt(self) -> None:
        """
        Print the content of all formatted prompts, if any.
        """
        if self.prompt is not None:
            for prompt_segment in self.prompt:
                print(prompt_segment["content"])


class LLMBase:
    """
    Base class for LLM wrappers, handling shared attributes (e.g., API key, model name).
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "gpt-4o-mini",
        timeout: int = 60,
        maximum_generation_attempts: int = 3,
        maximum_timeout_attempts: int = 3,
        debug: bool = False,
        azure: bool = False,
        azure_api_base: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            api_key: The LLM service API key.
            model: Which model to use (e.g., 'gpt-4o-mini').
            timeout: Max time limit (seconds) for a single request.
            maximum_generation_attempts: Maximum times to attempt generation.
            maximum_timeout_attempts: Maximum times to retry if timeouts occur.
            debug: Enable debug logs if True.
            azure: Whether the model is hosted on Azure.
            azure_api_base: The base URL for Azure endpoints (if azure=True).
            azure_api_version: The Azure API version (if azure=True).
            azure_deployment_name: The name of the Azure model deployment (if azure=True).
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.maximum_generation_attempts = maximum_generation_attempts
        self.maximum_timeout_attempts = maximum_timeout_attempts
        self.debug = debug
        self.azure = azure
        self.azure_api_base = azure_api_base
        self.azure_api_version = azure_api_version
        self.azure_deployment_name = azure_deployment_name


class OpenAI_interface(LLMBase):
    """
    A client for interacting with OpenAI's interface or Azure OpenAI service.

    This class provides methods to communicate with models through OpenAI's API or
    Azure OpenAI service, with built-in retry functionality for handling timeouts.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: int = 60,
        maximum_generation_attempts: int = 3,
        maximum_timeout_attempts: int = 3,
        debug: bool = False,
        azure: bool = False,
        azure_api_base: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        azure_deployment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the OpenAI or Azure OpenAI client.

        Args:
            api_key (str): The OpenAI API key for authentication.
            model (str): Model name (e.g., 'gpt-4o-mini').
            timeout (int): Max time limit (seconds) for a single request.
            maximum_generation_attempts (int): Max generation attempts.
            maximum_timeout_attempts (int): Max retries on timeouts.
            debug (bool): If True, print debug logs.
            azure (bool): If True, use Azure endpoints.
            azure_api_base (Optional[str]): Azure base URL.
            azure_api_version (Optional[str]): Azure API version.
            azure_deployment_name (Optional[str]): Azure model deployment name.
        """
        super().__init__(
            api_key,
            model,
            timeout,
            maximum_generation_attempts,
            maximum_timeout_attempts,
            debug,
            azure,
            azure_api_base,
            azure_api_version,
            azure_deployment_name,
        )

        # Initialize either a standard OpenAI client or an Azure-based one
        if self.azure:
            if not azure_api_base or not azure_deployment_name:
                raise ValueError(
                    "Azure API base and deployment name are required when using Azure."
                )
            try:
                # Attempt to import AzureOpenAI from the new sdk:
                try:
                    from openai import AzureOpenAI

                    self.client = AzureOpenAI(
                        api_key=api_key,
                        azure_endpoint=azure_api_base,
                        api_version=azure_api_version or "2023-12-01-preview",
                    )
                    print(f"Using AzureOpenAI client with endpoint: {azure_api_base}")
                except (ImportError, ModuleNotFoundError):
                    # Fallback to standard OpenAI with Azure settings
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url=f"{azure_api_base}/openai/deployments/{azure_deployment_name}",
                    )
                    print(
                        f"Using OpenAI client with base_url: "
                        f"{azure_api_base}/openai/deployments/{azure_deployment_name}"
                    )
            except Exception as e:
                print(f"Error initializing Azure OpenAI client: {e}")
                raise

            # Validate model compatibility with Azure
            supported_models = ["gpt-4o-mini", "gpt-4o", "deepseek-chat", "deepseek-reasoner"]
            if model not in supported_models:
                raise ValueError(
                    f"Model {model} is not supported for Azure. "
                    f"Supported models: {supported_models}"
                )

        elif self.model == "deepseek-chat" or self.model == "deepseek-reasoner":
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        else:
            self.client = OpenAI(api_key=api_key)

    @staticmethod
    def print_prompt(messages: list[ChatCompletionMessageParam]) -> None:
        """
        Print each segment of a message prompt.
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
            messages: The messages to send in OpenAI-like format.
            ret_dict: A dictionary used to capture return info if provided.

        Returns:
            (response_text, cost): The model's response text, and the approximate cost.
        """
        if self.debug:
            print("---Prompt beginning marker---")
            self.print_prompt(messages)
            print("---Prompt ending marker---")

        try:
            # Create the completion
            if self.azure:
                # For Azure, use deployment-based calls
                if hasattr(self.client, "chat"):
                    completion = self.client.chat.completions.create(
                        model=self.azure_deployment_name,
                        messages=messages,
                    )
                else:
                    completion = self.client.chat.completions.create(
                        deployment_id=self.azure_deployment_name,
                        messages=messages,
                    )
            else:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )

            if completion.choices[0].message.content is None:
                return None, 0.0

            response_text: str = completion.choices[0].message.content

            if self.debug:
                print("---Response beginning marker---")
                print(response_text)
                print("---Response ending marker---")

            # Skip cost for Azure or compute with the Calculator
            if self.azure:
                cost = 0.0
            elif self.model in ("deepseek-chat", "deepseek-reasoner"):
                calculator_instance = Calculator(self.model, messages, response_text)
                cost = calculator_instance.calculate_cost_DeepSeek()
            else:
                calculator_instance = Calculator(self.model, messages, response_text)
                cost = calculator_instance.calculate_cost_GPT()

            if ret_dict is not None:
                ret_dict["result"] = (response_text, cost)

            return response_text, cost

        except Exception as e:
            print(f"Error during API call: {e}")
            if ret_dict is not None:
                ret_dict["error"] = str(e)
            return None, 0.0

    def ask(
        self,
        messages: list[ChatCompletionMessageParam],
        ret_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[Optional[str], float]:
        """
        Send a message to the chat model with retry functionality for handling timeouts.
        If the call times out or fails, returns ("termination_signal", 0.0).

        Args:
            messages: The messages (OpenAI chat format).
            ret_dict: Optional dictionary to store output.

        Returns:
            (response_text, cost):
                response_text: The model's text response, or "termination_signal" on failure.
                cost: Approximate cost.
        """

        def target_function(ret_dict: dict[str, Any], *args: Any) -> None:
            messages = args[0]
            self.ask_base(messages, ret_dict=ret_dict)

        exceeded, result = retry_overtime_kill(
            target_function=target_function,
            target_function_args=(messages,),
            time_limit=self.timeout,
            maximum_retry=self.maximum_timeout_attempts,
            ret=True,
        )

        if not result or "result" not in result:
            return "termination_signal", 0.0

        # The target_function sets "result" in ret_dict
        response_value = result.get("result")
        if response_value is None:
            return "termination_signal", 0.0

        response_text, cost = response_value

        if not exceeded:
            return response_text, cost
        else:
            return "termination_signal", cost
