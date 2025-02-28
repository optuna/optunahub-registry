from __future__ import annotations

from copy import deepcopy
from typing import Any
from typing import Callable


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

    def conversation_prompting(
        self,
        sequence_assembler: Callable[[list[str], list[str]], str],
        current_input: str,
    ) -> str:
        """
        This method is a template for assembling the prompts for conversation-based methods
        for sequence generation. Conversation-based methods generate a new output sequence
        by interacting with the user and the LLM in a conversational manner.

        Formula:
            Y_i = f_lambda(Phi_i(X_<=i, Y<i))

        Explanation:
            - `Y_i`: The output sequence generated in the current (`i`th) iteration.
            - `f_lambda`: The LLM.
            - `Phi_i` (`sequence_assembler`): Combines:
                - `X_<=i`: Input sequences from all iterations (`inputs`).
                - `Y<i`: Output sequences from all previous iterations (`previous_outputs`).

        Args:
            sequence_assembler (callable):
                A function that takes all inputs and previous outputs and assembles
                the input for the current iteration.
            current_input (str):
                The raw current input sequence provided for the current iteration.

        Returns:
            str: The assembled input sequence for the current iteration.

        Example:
            >>> def example_sequence_assembler(inputs, past_outputs):
            ...     return " | ".join(inputs) + " | " + " | ".join(past_outputs)
            ...
            >>> processor = PromptBase()
            >>> processor.input_history = ["X0"]
            >>> processor.output_history = ["Y1", "Y2"]
            >>> current_input = "X_current"
            >>> assembled_input = processor.conversation_prompting(
            ...     sequence_assembler=example_sequence_assembler,
            ...     current_input=current_input
            ... )
            >>> print(assembled_input)
            X0 | X_current | Y1 | Y2
        """

        # Include the current input in the input history for evaluation
        inputs = deepcopy(self.input_history)
        inputs.append(current_input)

        # Use deepcopy to safely retrieve previous outputs
        previous_outputs = deepcopy(self.output_history)

        print("inputs:", inputs)
        print("previous_outputs:", previous_outputs)

        # Assemble the input using the provided assembler function
        assembled_input = sequence_assembler(inputs, previous_outputs)

        return assembled_input

    @staticmethod
    def sequence_assembler_default(inputs: list[str], previous_outputs: list[str]) -> str:
        """
        A default sequence assembler that concatenates the input and previous outputs for
        conversation-based methods with formatted separators and spacing.

        Args:
            inputs (list[str]): The input sequences from all iterations, including the current input.
            previous_outputs (list[str]): The output sequences from all previous iterations.

        Returns:
            str: The assembled input sequence for the current iteration in a nicely formatted string.

        Example:
            >>> inputs = ["What did I just say?", "Hi, my name is Tom"]
            >>> previous_outputs = ["Nice to meet you, Tom"]
            >>> result = PromptBase.sequence_assembler_default(inputs, previous_outputs)
            >>> print(result)
            ------ From the user: ------
            Hi, my name is Tom

            ------ Your response: ------
            Nice to meet you, Tom

            ------ From the user: ------
            What did I just say?
        """
        assembled_input = []

        # Combine inputs and outputs in conversation order
        for i, user_input in enumerate(inputs[:-1]):  # Include all inputs except the current
            if i > 0:
                assembled_input.append("\n------ From the user: ------")
            else:
                assembled_input.append("------ From the user: ------")
            assembled_input.append(f"\n{user_input}")

            if i < len(previous_outputs):  # Match response with user input if available
                assembled_input.append("\n------ Your response: ------")
                assembled_input.append(f"\n{previous_outputs[i]}")

        # Add the current user input
        if len(inputs) > 1:
            assembled_input.append("\n------ From the user: ------")
        else:
            assembled_input.append("------ From the user: ------")
        assembled_input.append(f"\n{inputs[-1]}")

        # Join the assembled input with newlines
        return "\n".join(assembled_input)
