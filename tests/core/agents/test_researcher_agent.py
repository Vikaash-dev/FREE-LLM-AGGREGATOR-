import pytest
import re
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.core.agents.researcher_agent import ResearcherAgent, _parse_path_from_description
from src.core.agent_structures import AgentConfig, TaskContext, TaskResult
from src.core.planning_structures import Task, TaskStatus, ProjectContext
from src.core.aggregator import LLMAggregator
from src.core.tools_registry import ToolsRegistry
from src.core.tool_interface import ToolOutput
from src.core.research_structures import ResearchQuery, KnowledgeChunk
# No need to import IntelligentResearchAssistant if its direct calls are within SOPs we mock

@pytest.fixture
def mock_llm_aggregator_for_researcher():
    return MagicMock(spec=LLMAggregator)

@pytest.fixture
def mock_tools_registry_for_researcher():
    return MagicMock(spec=ToolsRegistry)

@pytest.fixture
def researcher_agent_config():
    return AgentConfig(agent_id="researcher_001", role_name="GenericResearcher")

@pytest.fixture
def sample_task_for_researcher(request):
    task_description = getattr(request, "param", "Research the latest trends in AI.")
    return Task(task_id="task_research_1", description=task_description)

@pytest.fixture
def sample_project_context_for_researcher():
    return ProjectContext(project_id="proj_research_test", project_name="AI Trends Study")

@pytest.fixture
def sample_task_context_for_researcher(sample_task_for_researcher, sample_project_context_for_researcher):
    return TaskContext(
        current_task=sample_task_for_researcher,
        project_context=sample_project_context_for_researcher
    )

@pytest.fixture
def researcher_agent_with_mocked_sops(researcher_agent_config, mock_llm_aggregator_for_researcher, mock_tools_registry_for_researcher):
    # In ResearcherAgent, IntelligentResearchAssistant is created in __init__.
    # To fully mock SOPs, we'd also need to control IRA if SOPs call it directly.
    # However, for orchestrator tests, we mock the SOPs themselves.
    # If SOPs make calls to self.research_assistant, those won't happen if the SOP is mocked.

    # Patch IntelligentResearchAssistant during ResearcherAgent's instantiation for these tests
    with patch('src.core.agents.researcher_agent.IntelligentResearchAssistant', autospec=True) as MockIRA:
        # mock_ira_instance = MockIRA.return_value # This would be for testing IRA calls from SOPs

        agent = ResearcherAgent(
            researcher_agent_config,
            mock_llm_aggregator_for_researcher,
            mock_tools_registry_for_researcher
        )
        # Mock the SOP helper methods
        agent._sop_clarify_research_goal = AsyncMock(return_value="Clarified: " + researcher_agent_config.agent_id)
        agent._sop_execute_research_cycle = AsyncMock(return_value=([KnowledgeChunk(content="Chunk 1")], None)) # chunks, error
        agent._sop_format_report = MagicMock(return_value="Formatted Report from SOP mock.")
        agent._sop_save_report_to_file = AsyncMock(return_value=None) # error_string

        # Mock base methods if necessary (though SOPs should ideally encapsulate these)
        agent._use_llm = AsyncMock()
        agent._use_tool = AsyncMock()
        yield agent # Use yield to ensure patch is active for the test then cleaned up

# Test for the helper _parse_path_from_description (copied from agent, assuming it's defined there or imported)
def test_researcher_parse_path_from_description_found():
    text = "Please save report to 'output/report.md' about findings."
    assert _parse_path_from_description(text, "save report to") == "output/report.md"

@pytest.mark.asyncio
async def test_researcher_agent_sop_orchestration_success(researcher_agent_with_mocked_sops, sample_task_for_researcher, sample_task_context_for_researcher):
    agent = researcher_agent_with_mocked_sops
    original_desc = "Research AI and save report to 'report.txt'"
    sample_task_context_for_researcher.current_task.description = original_desc

    result = await agent.execute_task(sample_task_context_for_researcher.current_task, sample_task_context_for_researcher)

    agent._sop_clarify_research_goal.assert_called_once_with(original_desc, sample_task_context_for_researcher)
    agent._sop_execute_research_cycle.assert_called_once_with("Clarified: researcher_001", sample_task_context_for_researcher)
    agent._sop_format_report.assert_called_once_with([KnowledgeChunk(content="Chunk 1")])
    agent._sop_save_report_to_file.assert_called_once_with("Formatted Report from SOP mock.", original_desc, sample_task_context_for_researcher)

    assert result.status == TaskStatus.COMPLETED
    assert result.output["knowledge_chunks"] == [KnowledgeChunk(content="Chunk 1")]
    assert result.output["formatted_report"] == "Formatted Report from SOP mock."
    assert "report_file" in result.output # Because "save report" was in description

@pytest.mark.asyncio
async def test_researcher_agent_sop_research_cycle_error(researcher_agent_with_mocked_sops, sample_task_for_researcher, sample_task_context_for_researcher):
    agent = researcher_agent_with_mocked_sops
    agent._sop_execute_research_cycle.return_value = (None, "Mocked research cycle error")

    result = await agent.execute_task(sample_task_context_for_researcher.current_task, sample_task_context_for_researcher)

    assert result.status == TaskStatus.FAILED
    assert "Mocked research cycle error" in result.error_message
    agent._sop_format_report.assert_not_called()
    agent._sop_save_report_to_file.assert_not_called()

@pytest.mark.asyncio
async def test_researcher_agent_sop_research_yields_no_chunks(researcher_agent_with_mocked_sops, sample_task_for_researcher, sample_task_context_for_researcher):
    agent = researcher_agent_with_mocked_sops
    agent._sop_execute_research_cycle.return_value = ([], None) # Empty list of chunks

    result = await agent.execute_task(sample_task_context_for_researcher.current_task, sample_task_context_for_researcher)

    assert result.status == TaskStatus.COMPLETED # Task completed, but no info found
    assert result.output == [] # Empty list
    assert "Research yielded no specific information chunks" in result.error_message # Informative message
    agent._sop_format_report.assert_not_called() # Formatting might be skipped if no chunks
    agent._sop_save_report_to_file.assert_not_called()


@pytest.mark.asyncio
async def test_researcher_agent_sop_save_report_error(researcher_agent_with_mocked_sops, sample_task_for_researcher, sample_task_context_for_researcher):
    agent = researcher_agent_with_mocked_sops
    sample_task_context_for_researcher.current_task.description = "Research AI and save report to 'failed_report.txt'"
    agent._sop_save_report_to_file.return_value = "Mocked save report error" # error_string

    result = await agent.execute_task(sample_task_context_for_researcher.current_task, sample_task_context_for_researcher)

    assert result.status == TaskStatus.FAILED
    assert "Mocked save report error" in result.error_message
    assert result.output["knowledge_chunks"] == [KnowledgeChunk(content="Chunk 1")] # Research itself was fine
    assert result.output["formatted_report"] == "Formatted Report from SOP mock."


@pytest.mark.asyncio
async def test_researcher_agent_sop_no_save_instruction(researcher_agent_with_mocked_sops, sample_task_for_researcher, sample_task_context_for_researcher):
    agent = researcher_agent_with_mocked_sops
    sample_task_context_for_researcher.current_task.description = "Just research AI stuff." # No "save report"

    # Reset call count for _sop_save_report_to_file specifically for this test if agent is reused
    agent._sop_save_report_to_file.reset_mock()

    result = await agent.execute_task(sample_task_context_for_researcher.current_task, sample_task_context_for_researcher)

    agent._sop_clarify_research_goal.assert_called_once()
    agent._sop_execute_research_cycle.assert_called_once()
    agent._sop_format_report.assert_called_once()
    agent._sop_save_report_to_file.assert_called_once() # It's called, but will internally decide not to write

    assert result.status == TaskStatus.COMPLETED
    assert result.output["knowledge_chunks"] == [KnowledgeChunk(content="Chunk 1")]
    assert "report_file" not in result.output # Should not attempt to save if not instructed


# Test internal logic of an SOP step, e.g., _sop_save_report_to_file calling _use_tool
@pytest.mark.asyncio
async def test_researcher_agent_sop_save_report_calls_use_tool(
    researcher_agent_config, mock_llm_aggregator_for_researcher, mock_tools_registry_for_researcher,
    sample_task_context_for_researcher
):
    # This test does NOT mock the SOP methods on the agent instance.
    # It tests the actual _sop_save_report_to_file method.
    with patch('src.core.agents.researcher_agent.IntelligentResearchAssistant', autospec=True): # Still mock IRA
        agent = ResearcherAgent(
            researcher_agent_config,
            mock_llm_aggregator_for_researcher,
            mock_tools_registry_for_researcher
        )
        agent._use_tool = AsyncMock(return_value=ToolOutput(status_code=200)) # Successful write

        # Mock other SOPs to isolate the one being tested
        agent._sop_clarify_research_goal = AsyncMock(return_value="Clarified goal")
        agent._sop_execute_research_cycle = AsyncMock(return_value=([KnowledgeChunk(content="Test chunk")], None))
        agent._sop_format_report = MagicMock(return_value="Formatted test report content")

        task_desc_with_save = "Research things and save report to 'test_report.txt'"
        sample_task_context_for_researcher.current_task.description = task_desc_with_save

        await agent.execute_task(sample_task_context_for_researcher.current_task, sample_task_context_for_researcher)

        agent._use_tool.assert_called_once_with("file_write", {"file_path": "test_report.txt", "content": "Formatted test report content"})
