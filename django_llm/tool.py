"""Tool/function calling support for LLM interactions."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, get_type_hints


class Tool(ABC):
    """Base class for tools that can be called by LLMs.

    Subclass this and implement the `execute` method to create custom tools.

    Example:
        class WeatherTool(Tool):
            name = "get_weather"
            description = "Get the current weather for a location"

            def execute(self, location: str, unit: str = "celsius") -> str:
                # Fetch weather data...
                return f"Weather in {location}: 22{unit[0].upper()}"
    """

    # Override these in subclasses
    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with the given arguments.

        Returns:
            The result of the tool execution (will be converted to string).
        """
        pass

    def get_parameters(self) -> dict[str, Any]:
        """Get the JSON schema for this tool's parameters.

        Introspects the execute method to build the schema.
        """
        sig = inspect.signature(self.execute)
        hints = get_type_hints(self.execute)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "kwargs"):
                continue

            param_schema: dict[str, Any] = {}
            param_type = hints.get(param_name, str)

            # Map Python types to JSON schema types
            if param_type is str:
                param_schema["type"] = "string"
            elif param_type is int:
                param_schema["type"] = "integer"
            elif param_type is float:
                param_schema["type"] = "number"
            elif param_type is bool:
                param_schema["type"] = "boolean"
            elif param_type is list or (hasattr(param_type, "__origin__") and param_type.__origin__ is list):
                param_schema["type"] = "array"
            elif param_type is dict or (hasattr(param_type, "__origin__") and param_type.__origin__ is dict):
                param_schema["type"] = "object"
            else:
                param_schema["type"] = "string"

            properties[param_name] = param_schema

            # Check if parameter is required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def to_schema(self) -> dict[str, Any]:
        """Convert tool to schema format for API calls."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters(),
        }


class FunctionTool(Tool):
    """Create a tool from a regular function.

    Example:
        def calculate_sum(a: int, b: int) -> int:
            '''Add two numbers together.'''
            return a + b

        tool = FunctionTool(calculate_sum)
    """

    def __init__(self, func: Callable, name: str | None = None, description: str | None = None):
        self._func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or f"Execute {self.name}"

    def execute(self, **kwargs) -> Any:
        return self._func(**kwargs)

    def get_parameters(self) -> dict[str, Any]:
        """Get parameters from the wrapped function."""
        sig = inspect.signature(self._func)
        hints = get_type_hints(self._func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "kwargs"):
                continue

            param_schema: dict[str, Any] = {}
            param_type = hints.get(param_name, str)

            # Map Python types to JSON schema types
            if param_type is str:
                param_schema["type"] = "string"
            elif param_type is int:
                param_schema["type"] = "integer"
            elif param_type is float:
                param_schema["type"] = "number"
            elif param_type is bool:
                param_schema["type"] = "boolean"
            elif param_type is list:
                param_schema["type"] = "array"
            elif param_type is dict:
                param_schema["type"] = "object"
            else:
                param_schema["type"] = "string"

            properties[param_name] = param_schema

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


def tool(name: str | None = None, description: str | None = None) -> Callable:
    """Decorator to create a tool from a function.

    Example:
        @tool(description="Add two numbers together")
        def add(a: int, b: int) -> int:
            return a + b
    """
    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(func, name=name, description=description)
    return decorator
