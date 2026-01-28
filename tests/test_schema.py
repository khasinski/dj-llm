"""Tests for schema.py - structured output support."""

from dataclasses import dataclass

import pytest

from django_llm.schema import (
    _python_type_to_json_schema,
    parse_json_response,
    schema_to_json_schema,
)


class TestPythonTypeToJsonSchema:
    """Tests for _python_type_to_json_schema."""

    def test_string_type(self):
        assert _python_type_to_json_schema(str) == {"type": "string"}

    def test_int_type(self):
        assert _python_type_to_json_schema(int) == {"type": "integer"}

    def test_float_type(self):
        assert _python_type_to_json_schema(float) == {"type": "number"}

    def test_bool_type(self):
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list_type(self):
        assert _python_type_to_json_schema(list) == {"type": "array"}

    def test_dict_type(self):
        assert _python_type_to_json_schema(dict) == {"type": "object"}

    def test_generic_list_type(self):
        result = _python_type_to_json_schema(list[str])
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_generic_list_int_type(self):
        result = _python_type_to_json_schema(list[int])
        assert result == {"type": "array", "items": {"type": "integer"}}


class TestSchemaToJsonSchema:
    """Tests for schema_to_json_schema."""

    def test_dict_passthrough(self):
        """Dict schemas are passed through with additionalProperties."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = schema_to_json_schema(schema)
        assert result["type"] == "object"
        assert result["additionalProperties"] is False

    def test_dict_preserves_existing_additional_properties(self):
        """Existing additionalProperties is preserved."""
        schema = {"type": "object", "additionalProperties": True}
        result = schema_to_json_schema(schema)
        assert result["additionalProperties"] is True

    def test_dataclass_conversion(self):
        """Dataclasses are converted to JSON schema."""

        @dataclass
        class Person:
            name: str
            age: int

        result = schema_to_json_schema(Person)
        assert result["type"] == "object"
        assert "name" in result["properties"]
        assert "age" in result["properties"]
        assert result["properties"]["name"] == {"type": "string"}
        assert result["properties"]["age"] == {"type": "integer"}
        assert result["additionalProperties"] is False

    def test_dataclass_required_fields(self):
        """All fields are required for OpenAI strict mode."""

        @dataclass
        class Person:
            name: str
            age: int
            occupation: str

        result = schema_to_json_schema(Person)
        assert set(result["required"]) == {"name", "age", "occupation"}

    def test_dataclass_with_complex_types(self):
        """Dataclasses with complex types are converted."""

        @dataclass
        class Data:
            tags: list[str]
            active: bool
            score: float

        result = schema_to_json_schema(Data)
        assert result["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
        assert result["properties"]["active"] == {"type": "boolean"}
        assert result["properties"]["score"] == {"type": "number"}

    def test_invalid_schema_raises_error(self):
        """Non-dataclass, non-dict raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            schema_to_json_schema("invalid")


class TestParseJsonResponse:
    """Tests for parse_json_response."""

    def test_parse_plain_json(self):
        """Plain JSON is parsed."""
        result = parse_json_response('{"name": "John", "age": 30}')
        assert result == {"name": "John", "age": 30}

    def test_parse_json_with_markdown_block(self):
        """JSON in markdown code block is parsed."""
        content = '```json\n{"name": "John"}\n```'
        result = parse_json_response(content)
        assert result == {"name": "John"}

    def test_parse_json_with_plain_markdown_block(self):
        """JSON in plain markdown code block is parsed."""
        content = '```\n{"name": "John"}\n```'
        result = parse_json_response(content)
        assert result == {"name": "John"}

    def test_parse_into_dataclass(self):
        """JSON is parsed into a dataclass instance."""

        @dataclass
        class Person:
            name: str
            age: int

        result = parse_json_response('{"name": "John", "age": 30}', Person)
        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30

    def test_parse_without_schema_returns_dict(self):
        """Without schema, dict is returned."""
        result = parse_json_response('{"key": "value"}', None)
        assert result == {"key": "value"}


class TestPydanticIntegration:
    """Tests for Pydantic model support (if available)."""

    def test_pydantic_v2_schema(self):
        """Pydantic v2 models are supported."""
        try:
            from pydantic import BaseModel

            class Person(BaseModel):
                name: str
                age: int

            result = schema_to_json_schema(Person)
            assert result["type"] == "object"
            assert "name" in result["properties"]
            assert result["additionalProperties"] is False
        except ImportError:
            pytest.skip("Pydantic not installed")

    def test_pydantic_v2_parse(self):
        """Pydantic v2 models can be parsed."""
        try:
            from pydantic import BaseModel

            class Person(BaseModel):
                name: str
                age: int

            result = parse_json_response('{"name": "John", "age": 30}', Person)
            assert isinstance(result, Person)
            assert result.name == "John"
            assert result.age == 30
        except ImportError:
            pytest.skip("Pydantic not installed")
