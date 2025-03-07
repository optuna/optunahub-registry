from __future__ import annotations

from typing import Any


class PromptBase:
    """
    A base class for managing and formatting prompts for Language Learning Models (LLMs).

    This class provides functionality to store, format, and display prompts in a format
    compatible with GPT-style language models. It maintains an internal list of formatted
    prompts and provides methods to manipulate and view them.

    Attributes:
        prompt (Any): The formatted prompts ready for being passed to the LLM.
        conversation_history (list[str]): The history of conversations recorded.

    Example:
        >>> prompt_manager = PromptBase()
        >>> formatted = prompt_manager.list_to_formatted_OpenAI(["Hello", "World"])
        >>> len(formatted)
        2
        >>> formatted[0]["content"]
        'Hello'
        >>> formatted[1]["content"]
        '\\nWorld'
    """

    def __init__(self) -> None:
        """
        Initialize a new PromptBase instance with an empty prompt list and conversation history.

        Example:
            >>> prompt_base = PromptBase()
            >>> prompt_base.prompt
            None
            >>> prompt_base.conversation_history
            []
        """
        self.prompt: Any = None
        self.input_history: list[str] = []
        self.output_history: list[str] = []
        self.conversation_history: list[str] = []

    @staticmethod
    def list_to_formatted_OpenAI(prompt_as_list: list[str]) -> list[dict[str, str]]:
        """
        Format a list of prompt strings into formats compatible with OpenAI interface.

        This method takes a list of strings and converts them into the format expected
        by OpenAI-style models, where all prompt segments are combined into a single content
        string, separated by newline characters.

        Args:
            prompt_as_list (list[str]): A list of strings representing the prompt segments to
                be formatted.

        Returns:
            list[dict[str, str]]: A list of dictionaries where each dictionary contains
                'role' and 'content' keys. The list will contain only one dictionary.

        Example:
            >>> base = PromptBase()
            >>> result = base.list_to_formatted_OpenAI(["First prompt", "Second prompt"])
            >>> result[0]["content"]
            'First prompt\nSecond prompt'
        """
        combined_content = "\n".join(prompt_as_list)
        formatted_prompt = [{"role": "user", "content": combined_content}]
        return formatted_prompt

    @staticmethod
    def formatted_to_string_OpenAI(formatted_prompt: list[dict[str, str]]) -> str:
        """
        Convert a formatted prompt (list of dictionaries with 'role' and 'content' keys)
        into a single concatenated string.

        This method takes a list of prompt segments in the OpenAI format and combines them
        into a single string. The 'role' is ignored, and only the 'content' is used.

        Args:
            formatted_prompt (list[dict[str, str]]): A list of dictionaries where each
                dictionary contains 'role' and 'content' keys.

        Returns:
            str: A single concatenated string of all the 'content' values.

        Example:
            >>> base = PromptBase()
            >>> formatted_prompt = [
            ...     {"role": "system", "content": "First prompt"},
            ...     {"role": "user", "content": "Second prompt"}
            ... ]
            >>> result = base.formatted_to_string_OpenAI(formatted_prompt)
            >>> print(result)
            'First prompt\\nSecond prompt'
        """
        concatenated_string = ""
        for segment in formatted_prompt:
            concatenated_string += segment["content"]
        return concatenated_string

    def print_prompt(self) -> None:
        """
        Print the content of all formatted prompts.

        This method iterates through the stored prompts and prints their content
        to standard output.

        Example:
            >>> base = PromptBase()
            >>> base.prompt = [{"role": "system", "content": "Test prompt"}]
            >>> base.print_prompt()
            Test prompt
        """
        if self.prompt is not None:
            for prompt_segment in self.prompt:
                print(prompt_segment["content"])
