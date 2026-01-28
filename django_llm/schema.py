"""Structured output schema support."""

import json
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar, get_type_hints

T = TypeVar("T")


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    """Convert a Python type to JSON schema."""
    if py_type is str:
        return {"type": "string"}
    elif py_type is int:
        return {"type": "integer"}
    elif py_type is float:
        return {"type": "number"}
    elif py_type is bool:
        return {"type": "boolean"}
    elif py_type is list:
        return {"type": "array"}
    elif py_type is dict:
        return {"type": "object"}
    elif hasattr(py_type, "__origin__"):
        # Handle generics like list[str], dict[str, int]
        origin = py_type.__origin__
        if origin is list:
            args = getattr(py_type, "__args__", (str,))
            return {
                "type": "array",
                "items": _python_type_to_json_schema(args[0]) if args else {"type": "string"},
            }
        elif origin is dict:
            return {"type": "object"}
    return {"type": "string"}


def schema_to_json_schema(schema: type[T] | dict[str, Any]) -> dict[str, Any]:
    """Convert a schema (dataclass, Pydantic model, or dict) to JSON schema.

    Args:
        schema: A dataclass, Pydantic BaseModel subclass, or raw JSON schema dict.

    Returns:
        JSON schema dict compatible with OpenAI/Anthropic/Google APIs.
    """
    # Already a dict schema
    if isinstance(schema, dict):
        result = schema.copy()
        # Ensure additionalProperties is set for OpenAI strict mode
        if "additionalProperties" not in result:
            result["additionalProperties"] = False
        return result

    # Check for Pydantic model
    if hasattr(schema, "model_json_schema"):
        # Pydantic v2
        result = schema.model_json_schema()
        if "additionalProperties" not in result:
            result["additionalProperties"] = False
        return result
    elif hasattr(schema, "schema"):
        # Pydantic v1
        result = schema.schema()
        if "additionalProperties" not in result:
            result["additionalProperties"] = False
        return result

    # Dataclass
    if is_dataclass(schema):
        hints = get_type_hints(schema)
        properties = {}

        for field in fields(schema):
            field_type = hints.get(field.name, str)
            properties[field.name] = _python_type_to_json_schema(field_type)

        # OpenAI strict mode requires all properties in required
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "additionalProperties": False,
        }

    raise ValueError(f"Cannot convert {schema} to JSON schema. Use a dataclass, Pydantic model, or dict.")


def parse_json_response(content: str, schema: type[T] | None = None) -> T | dict[str, Any]:
    """Parse a JSON response, optionally into a schema type.

    Args:
        content: The JSON string from the LLM response.
        schema: Optional schema class to instantiate.

    Returns:
        Parsed data as dict or schema instance.
    """
    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        # Remove first and last lines (```json and ```)
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```"):
                in_block = not in_block
                continue
            if in_block:
                json_lines.append(line)
        content = "\n".join(json_lines)

    data = json.loads(content)

    if schema is None:
        return data

    # Pydantic model
    if hasattr(schema, "model_validate"):
        return schema.model_validate(data)
    elif hasattr(schema, "parse_obj"):
        return schema.parse_obj(data)

    # Dataclass
    if is_dataclass(schema):
        return schema(**data)

    return data
