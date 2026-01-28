"""Tests for the Tool class."""

from django_llm.tool import FunctionTool, Tool, tool


class WeatherTool(Tool):
    """Example tool for testing."""

    name = "get_weather"
    description = "Get the current weather for a location"

    def execute(self, location: str, unit: str = "celsius") -> str:
        return f"Weather in {location}: 22 {unit}"


class CalculatorTool(Tool):
    """Example calculator tool."""

    name = "calculate"
    description = "Perform basic arithmetic"

    def execute(self, operation: str, a: int, b: int) -> int:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            return a // b
        raise ValueError(f"Unknown operation: {operation}")


class TestTool:
    """Tests for Tool class."""

    def test_tool_execution(self):
        """Test basic tool execution."""
        tool = WeatherTool()
        result = tool.execute(location="New York")
        assert "New York" in result
        assert "22" in result

    def test_tool_with_default_args(self):
        """Test tool execution with default arguments."""
        tool = WeatherTool()
        result = tool.execute(location="London", unit="fahrenheit")
        assert "London" in result
        assert "fahrenheit" in result

    def test_tool_schema(self):
        """Test tool schema generation."""
        tool = WeatherTool()
        schema = tool.to_schema()

        assert schema["name"] == "get_weather"
        assert schema["description"] == "Get the current weather for a location"
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
        assert "location" in schema["parameters"]["properties"]
        assert "unit" in schema["parameters"]["properties"]
        assert "location" in schema["parameters"]["required"]
        assert "unit" not in schema["parameters"]["required"]

    def test_calculator_tool(self):
        """Test calculator tool."""
        tool = CalculatorTool()
        assert tool.execute(operation="add", a=5, b=3) == 8
        assert tool.execute(operation="multiply", a=4, b=7) == 28

    def test_parameter_types(self):
        """Test that parameter types are correctly inferred."""
        tool = CalculatorTool()
        params = tool.get_parameters()

        assert params["properties"]["operation"]["type"] == "string"
        assert params["properties"]["a"]["type"] == "integer"
        assert params["properties"]["b"]["type"] == "integer"


class TestFunctionTool:
    """Tests for FunctionTool wrapper."""

    def test_function_tool_from_function(self):
        """Test creating a tool from a regular function."""
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        tool = FunctionTool(greet)
        assert tool.name == "greet"
        assert "Greet someone" in tool.description

        result = tool.execute(name="Alice")
        assert result == "Hello, Alice!"

    def test_function_tool_custom_name(self):
        """Test creating a tool with custom name."""
        def my_func(x: int) -> int:
            return x * 2

        tool = FunctionTool(my_func, name="doubler", description="Doubles a number")
        assert tool.name == "doubler"
        assert tool.description == "Doubles a number"


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator(self):
        """Test creating a tool with the decorator."""
        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert add.name == "add"
        assert add.description == "Add two numbers"
        assert add.execute(a=3, b=4) == 7

    def test_tool_decorator_with_name(self):
        """Test decorator with custom name."""
        @tool(name="summer", description="Sum numbers")
        def add_numbers(a: int, b: int) -> int:
            return a + b

        assert add_numbers.name == "summer"
