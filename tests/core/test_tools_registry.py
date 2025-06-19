import pytest
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel, ValidationError, Field as PydanticField # Alias Field to avoid pytest conflict

from src.core.tools_registry import ToolsRegistry
from src.core.tool_interface import ToolInterface, ToolOutput
from src.core.tools.file_system_tools import ReadFileTool # For a concrete tool example

# Define a MockTool for testing
class MockToolInput(BaseModel):
    param: str = PydanticField(description="A test parameter")
    optional_param: int = 0

class MockToolOutputSpecific(ToolOutput): # Example if a tool had a more specific output model
    specific_field: str = "default"

class MockTool(ToolInterface):
    name_val = "mock_tool"
    description_val = "A mock tool for testing."
    input_schema_val = MockToolInput
    output_schema_val = ToolOutput # Using generic ToolOutput for this mock

    @property
    def name(self) -> str:
        return self.name_val
    @property
    def description(self) -> str:
        return self.description_val
    @property
    def input_schema(self) -> type[BaseModel]:
        return self.input_schema_val
    @property
    def output_schema(self) -> type[ToolOutput]:
        return self.output_schema_val

    async def execute(self, inputs: MockToolInput) -> ToolOutput: # inputs type is MockToolInput
        if inputs.param == "error_execute":
            raise ValueError("Mock tool execution error")
        if inputs.param == "specific_output":
             # This mock uses generic ToolOutput, but if it used MockToolOutputSpecific:
             # return MockToolOutputSpecific(output=f"mock output for {inputs.param}", specific_field="custom")
            return ToolOutput(output=f"mock output for {inputs.param} with optional {inputs.optional_param}")
        return ToolOutput(output=f"mock output for {inputs.param} with optional {inputs.optional_param}")

@pytest.fixture
def tools_registry():
    return ToolsRegistry()

@pytest.fixture
def mock_tool_instance():
    return MockTool()

def test_register_and_get_tool(tools_registry: ToolsRegistry, mock_tool_instance: MockTool):
    tools_registry.register_tool(mock_tool_instance)
    assert tools_registry.get_tool("mock_tool") == mock_tool_instance
    assert tools_registry.get_tool("non_existent_tool") is None

def test_register_invalid_tool_type(tools_registry: ToolsRegistry):
    class NotATool:
        pass
    with pytest.raises(TypeError, match="Tool must implement ToolInterface"):
        tools_registry.register_tool(NotATool()) # type: ignore

def test_unregister_tool(tools_registry: ToolsRegistry, mock_tool_instance: MockTool):
    tools_registry.register_tool(mock_tool_instance)
    tools_registry.unregister_tool("mock_tool")
    assert tools_registry.get_tool("mock_tool") is None
    # Test unregistering non-existent (should not raise error, just log)
    tools_registry.unregister_tool("non_existent_tool")

@pytest.mark.asyncio
async def test_invoke_tool_success(tools_registry: ToolsRegistry, mock_tool_instance: MockTool):
    tools_registry.register_tool(mock_tool_instance)
    result = await tools_registry.invoke_tool("mock_tool", {"param": "test", "optional_param": 5})
    assert result.succeeded is True
    assert result.output == "mock output for test with optional 5"
    assert result.status_code == 200
    assert result.error is None

@pytest.mark.asyncio
async def test_invoke_tool_not_found(tools_registry: ToolsRegistry):
    result = await tools_registry.invoke_tool("non_existent_tool", {"param": "test"})
    assert result.succeeded is False
    assert result.error == "Tool 'non_existent_tool' not found."
    assert result.status_code == 404

@pytest.mark.asyncio
async def test_invoke_tool_input_validation_error(tools_registry: ToolsRegistry, mock_tool_instance: MockTool):
    tools_registry.register_tool(mock_tool_instance)
    # Missing 'param', which is required by MockToolInput
    result = await tools_registry.invoke_tool("mock_tool", {"wrong_param": "test"})
    assert result.succeeded is False
    assert "Input validation error for tool 'mock_tool'" in result.error # type: ignore
    assert "param\n  Field required" in result.error or "Field required" in result.error # Pydantic v2 vs v1
    assert result.status_code == 400

@pytest.mark.asyncio
async def test_invoke_tool_execution_error(tools_registry: ToolsRegistry, mock_tool_instance: MockTool):
    tools_registry.register_tool(mock_tool_instance)
    result = await tools_registry.invoke_tool("mock_tool", {"param": "error_execute"})
    assert result.succeeded is False
    assert "Error executing tool 'mock_tool': Mock tool execution error" in result.error # type: ignore
    assert result.status_code == 500

def test_list_tools(tools_registry: ToolsRegistry, mock_tool_instance: MockTool):
    assert tools_registry.list_tools() == []
    tools_registry.register_tool(mock_tool_instance)
    read_tool = ReadFileTool() # Another tool for testing list
    tools_registry.register_tool(read_tool)

    listed_tools = tools_registry.list_tools()
    assert len(listed_tools) == 2
    # Order might vary, so check for presence
    assert {"name": "mock_tool", "description": "A mock tool for testing."} in listed_tools
    assert {"name": "file_read", "description": "Reads the entire content of a specified file and returns it as a string."} in listed_tools

def test_get_tool_schema(tools_registry: ToolsRegistry, mock_tool_instance: MockTool):
    tools_registry.register_tool(mock_tool_instance)
    schema = tools_registry.get_tool_schema("mock_tool")
    assert schema == MockToolInput
    assert tools_registry.get_tool_schema("non_existent_tool") is None

def test_get_tool_output_def(tools_registry: ToolsRegistry, mock_tool_instance: MockTool):
    tools_registry.register_tool(mock_tool_instance)
    output_def = tools_registry.get_tool_output_def("mock_tool")
    assert output_def == ToolOutput # As defined in MockTool
    assert tools_registry.get_tool_output_def("non_existent_tool") is None
