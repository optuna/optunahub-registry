"""LLM package initialization."""

from typing import List

# Import all modules
from . import cost
from . import fault_tolerance
from . import inquiry
from . import prompter


# Initialize __all__ as an empty list
__all__: List[str] = []
# Extend __all__ with each module's __all__ (if available)
if hasattr(cost, "__all__"):
    __all__.extend(cost.__all__)
if hasattr(fault_tolerance, "__all__"):
    __all__.extend(fault_tolerance.__all__)
if hasattr(inquiry, "__all__"):
    __all__.extend(inquiry.__all__)
if hasattr(prompter, "__all__"):
    __all__.extend(prompter.__all__)
