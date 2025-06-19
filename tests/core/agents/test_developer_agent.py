import pytest
import re
from unittest.mock import AsyncMock, MagicMock, call

from src.core.agents.developer_agent import DeveloperAgent, _parse_path_from_description
from src.core.agent_structures import AgentConfig, TaskContext, TaskResult
from src.core.planning_structures import Task, TaskStatus, ProjectContext
from src.core.aggregator import LLMAggregator
from src.core.tools_registry import ToolsRegistry
from src.core.tool_interface import ToolOutput
from src.models import ChatMessage, ChatCompletionResponse, Choice, Message

@pytest.fixture
def mock_llm_aggregator_for_dev():
    return MagicMock(spec=LLMAggregator)

@pytest.fixture
def mock_tools_registry_for_dev():
    return MagicMock(spec=ToolsRegistry)

@pytest.fixture
def dev_agent_config():
    return AgentConfig(agent_id="dev_001", role_name="PythonDeveloper")

@pytest.fixture
def sample_task_for_dev(request):
    task_description = getattr(request, "param", "Write a Python function to add two numbers.")
    return Task(task_id="task_dev_1", description=task_description)

@pytest.fixture
def sample_task_context_for_dev(sample_task_for_dev):
    project_ctx = ProjectContext(project_id="proj_dev_test", project_name="Coding Project")
    return TaskContext(current_task=sample_task_for_dev, project_context=project_ctx)

@pytest.fixture
def developer_agent_with_mocked_sops(dev_agent_config, mock_llm_aggregator_for_dev, mock_tools_registry_for_dev):
    agent = DeveloperAgent(dev_agent_config, mock_llm_aggregator_for_dev, mock_tools_registry_for_dev)
    agent._sop_analyze_requirements = AsyncMock(return_value="Analyzed: " + dev_agent_config.agent_id) # Suffix to check it's from mock
    agent._sop_read_initial_files = AsyncMock(return_value=("", None)) # content, error
    agent._sop_generate_code = AsyncMock(return_value=("print('Hello from SOP')", None)) # code, error
    agent._sop_self_critique = AsyncMock(return_value=("print('Hello from critiqued SOP')", None)) # code, error
    agent._sop_write_output_files = AsyncMock(return_value=None) # error_string

    # Also mock base methods that SOPs might call, if not testing SOP internals here
    agent._use_llm = AsyncMock()
    agent._use_tool = AsyncMock()
    return agent

# Tests for _parse_path_from_description
def test_parse_path_from_description_found():
    text = "Please read file 'path/to/my_document.txt' and summarize it."
    assert _parse_path_from_description(text, "read file") == "path/to/my_document.txt"

def test_parse_path_from_description_not_found():
    text = "Read the specified document."
    assert _parse_path_from_description(text, "read file") is None

@pytest.mark.asyncio
async def test_developer_agent_sop_orchestration_success(developer_agent_with_mocked_sops, sample_task_for_dev, sample_task_context_for_dev):
    agent = developer_agent_with_mocked_sops
    sample_task_context_for_dev.current_task.description = "Develop a feature." # Simple description

    result = await agent.execute_task(sample_task_context_for_dev.current_task, sample_task_context_for_dev)

    agent._sop_analyze_requirements.assert_called_once_with("Develop a feature.", sample_task_context_for_dev)
    agent._sop_read_initial_files.assert_called_once_with("Develop a feature.", sample_task_context_for_dev)
    # Requirements come from _sop_analyze_requirements
    agent._sop_generate_code.assert_called_once_with("Analyzed: dev_001", "", sample_task_context_for_dev)
    # Code for critique comes from _sop_generate_code
    agent._sop_self_critique.assert_called_once_with("print('Hello from SOP')", "Analyzed: dev_001", sample_task_context_for_dev)
    # Code for writing comes from _sop_self_critique
    agent._sop_write_output_files.assert_called_once_with("print('Hello from critiqued SOP')", "Develop a feature.", sample_task_context_for_dev)

    assert result.status == TaskStatus.COMPLETED
    assert "Analyzed Requirements: Analyzed: dev_001" in result.output
    assert "Generated Code/Text:\nprint('Hello from SOP')" in result.output
    assert "Self-Critiqued Code:\nprint('Hello from critiqued SOP')" in result.output # Or "No changes" if critique doesn't change it

@pytest.mark.asyncio
async def test_developer_agent_sop_read_files_error(developer_agent_with_mocked_sops, sample_task_for_dev, sample_task_context_for_dev):
    agent = developer_agent_with_mocked_sops
    agent._sop_read_initial_files.return_value = ("", "Mocked file read error") # content, error

    result = await agent.execute_task(sample_task_context_for_dev.current_task, sample_task_context_for_dev)

    # Even with read error, other steps proceed but the error is noted.
    # The overall status might not be FAILED if reading is optional.
    # Current SOP treats it as a note, not a hard failure.
    assert result.status == TaskStatus.COMPLETED
    assert "File Reading Note: Mocked file read error" in result.output
    agent._sop_generate_code.assert_called_once() # Generate should still be called

@pytest.mark.asyncio
async def test_developer_agent_sop_generate_code_error(developer_agent_with_mocked_sops, sample_task_for_dev, sample_task_context_for_dev):
    agent = developer_agent_with_mocked_sops
    agent._sop_generate_code.return_value = ("Error: Code gen explosion", "Mocked code gen error")

    result = await agent.execute_task(sample_task_context_for_dev.current_task, sample_task_context_for_dev)

    assert result.status == TaskStatus.FAILED
    assert "Code Generation Error: Mocked code gen error" in result.output
    agent._sop_self_critique.assert_not_called() # Self-critique should be skipped
    agent._sop_write_output_files.assert_not_called() # Writing output should be skipped

@pytest.mark.asyncio
async def test_developer_agent_sop_self_critique_error_does_not_fail_task(developer_agent_with_mocked_sops, sample_task_for_dev, sample_task_context_for_dev):
    agent = developer_agent_with_mocked_sops
    # Code generation is successful
    generated_code = "print('Initial code')"
    agent._sop_generate_code.return_value = (generated_code, None)
    # Self-critique step simulates an error
    agent._sop_self_critique.return_value = (generated_code, "Mocked self-critique error")

    result = await agent.execute_task(sample_task_context_for_dev.current_task, sample_task_context_for_dev)

    assert result.status == TaskStatus.COMPLETED # Task still completes with un-critiqued code
    assert "Self-Critique Error: Mocked self-critique error" in result.output
    assert f"Generated Code/Text:\n{generated_code}" in result.output # Original generated code is present
    # Ensure that the code for writing is the un-critiqued one
    agent._sop_write_output_files.assert_called_once_with(generated_code, sample_task_for_dev.description, sample_task_context_for_dev)


@pytest.mark.asyncio
async def test_developer_agent_sop_write_output_files_error(developer_agent_with_mocked_sops, sample_task_for_dev, sample_task_context_for_dev):
    agent = developer_agent_with_mocked_sops
    # Assume task description implies writing a file
    sample_task_context_for_dev.current_task.description = "Generate code and write file 'output.py'"
    agent._sop_write_output_files.return_value = "Mocked file write error" # error_string

    result = await agent.execute_task(sample_task_context_for_dev.current_task, sample_task_context_for_dev)

    assert result.status == TaskStatus.FAILED # If writing was requested and failed, task fails
    assert "File Writing Error: Mocked file write error" in result.output

# To test internal logic of a specific SOP step (e.g., _sop_read_initial_files actually calling _use_tool)
# we would not mock that SOP step, but mock _use_tool instead.
@pytest.mark.asyncio
async def test_developer_agent_sop_read_initial_files_calls_use_tool(dev_agent_config, mock_llm_aggregator_for_dev, mock_tools_registry_for_dev, sample_task_context_for_dev):
    # Don't mock SOP methods, test their internals.
    agent = DeveloperAgent(dev_agent_config, mock_llm_aggregator_for_dev, mock_tools_registry_for_dev)
    agent._use_tool = AsyncMock(return_value=ToolOutput(output="file content", status_code=200))

    # Mock other SOPs that are not the focus of this test to isolate _sop_read_initial_files
    agent._sop_analyze_requirements = AsyncMock(return_value="Analyzed requirements")
    agent._sop_generate_code = AsyncMock(return_value=("Generated code", None))
    agent._sop_self_critique = AsyncMock(return_value=("Critiqued code", None))
    agent._sop_write_output_files = AsyncMock(return_value=None)

    task_desc_with_read = "read file 'test.py' and then do something"
    sample_task_context_for_dev.current_task.description = task_desc_with_read

    await agent.execute_task(sample_task_context_for_dev.current_task, sample_task_context_for_dev)

    agent._use_tool.assert_called_once_with("file_read", {"file_path": "test.py"})
    # Check that generate_code received the file content
    agent._sop_generate_code.assert_called_once_with("Analyzed requirements", "file content", sample_task_context_for_dev)

@pytest.mark.asyncio
async def test_developer_agent_sop_write_output_files_calls_use_tool(dev_agent_config, mock_llm_aggregator_for_dev, mock_tools_registry_for_dev, sample_task_context_for_dev):
    agent = DeveloperAgent(dev_agent_config, mock_llm_aggregator_for_dev, mock_tools_registry_for_dev)
    agent._use_tool = AsyncMock(return_value=ToolOutput(status_code=200)) # Successful write

    # Mock other SOPs
    agent._sop_analyze_requirements = AsyncMock(return_value="Analyzed requirements")
    agent._sop_read_initial_files = AsyncMock(return_value=("", None))
    mock_generated_code = "print('final code')"
    agent._sop_generate_code = AsyncMock(return_value=(mock_generated_code, None))
    agent._sop_self_critique = AsyncMock(return_value=(mock_generated_code, None)) # Assume critique makes no change

    task_desc_with_write = "generate code and write file 'out.py'"
    sample_task_context_for_dev.current_task.description = task_desc_with_write

    await agent.execute_task(sample_task_context_for_dev.current_task, sample_task_context_for_dev)

    agent._use_tool.assert_called_once_with("file_write", {"file_path": "out.py", "content": mock_generated_code})
