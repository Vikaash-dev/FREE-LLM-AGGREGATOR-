from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Type as TypingType # Renamed Type to TypingType to avoid conflict

from pydantic import BaseModel, Field


class ToolOutput(BaseModel):
    """
    Standardized output structure for all tools.
    """
    output: Optional[Any] = None
    error: Optional[str] = None
    status_code: int = 200  # HTTP-like status codes: 200 for success, 4xx for client errors, 5xx for server/tool errors

    @property
    def succeeded(self) -> bool:
        return self.error is None and 200 <= self.status_code < 300


class ToolInterface(ABC):
    """
    Abstract base class defining the interface for all tools.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A brief description of what the tool does."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> TypingType[BaseModel]:
        """The Pydantic model defining the input schema for the tool."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> TypingType[ToolOutput]: # Typically ToolOutput or a subclass
        """
        The Pydantic model defining the output schema for the tool.
        Usually, this will be ToolOutput itself or a more specific subclass if needed.
        """
        pass

    @abstractmethod
    async def execute(self, inputs: BaseModel) -> ToolOutput:
        """
        Executes the tool with the given validated inputs.

        Args:
            inputs: A Pydantic model instance conforming to `self.input_schema`.

        Returns:
            A ToolOutput object representing the result of the execution.
        """
        pass
