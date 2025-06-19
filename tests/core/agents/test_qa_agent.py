import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from src.core.agents.qa_agent import QAAgent
from src.core.agent_structures import AgentConfig, TaskContext, TaskResult
from src.core.planning_structures import Task, TaskStatus, ProjectContext
from src.core.aggregator import LLMAggregator
from src.core.tools_registry import ToolsRegistry
from src.models import ChatMessage, ChatCompletionResponse, Choice as ChatCompletionChoice, Message, Usage as ChatCompletionUsage


@pytest.fixture
def mock_llm_aggregator_for_qa():
    return MagicMock(spec=LLMAggregator)

@pytest.fixture
def mock_tools_registry_for_qa():
    return MagicMock(spec=ToolsRegistry)

@pytest.fixture
def qa_agent_config():
    return AgentConfig(agent_id="qa_agent_01", role_name="QualityAssuranceAgent")

@pytest.fixture
def qa_agent(qa_agent_config, mock_llm_aggregator_for_qa, mock_tools_registry_for_qa):
    agent = QAAgent(
        agent_config=qa_agent_config,
        llm_aggregator=mock_llm_aggregator_for_qa,
        tools_registry=mock_tools_registry_for_qa
    )
    # Mock the _use_llm method for direct control over its return value in tests
    agent._use_llm = AsyncMock()
    return agent

@pytest.mark.asyncio
async def test_qa_agent_execute_task_success(qa_agent, qa_agent_config):
    content_to_review = "This is a sample text that needs quality assurance."
    criteria = "Check for clarity, conciseness, and grammatical correctness."
    # Simulate task.raw_instruction containing the details as a JSON string
    raw_instruction_dict = {"content_to_qa": content_to_review, "qa_criteria": criteria}

    task = Task(
        task_id="qa_task_1",
        description="Perform QA on the provided text.",
        raw_instruction=json.dumps(raw_instruction_dict)
    )
    context = TaskContext(current_task=task)

    mock_llm_output = {
        "overall_assessment": "Approved with minor revisions",
        "issues_found": [{"issue_id": "ISSUE-001", "description": "Sentence in paragraph 2 is a bit lengthy.", "severity": "Low", "location_suggestion": "Paragraph 2", "suggested_fix": "Split the sentence."}],
        "suggestions_for_improvement": ["Consider adding a concluding summary."],
        "confidence_score": 0.85
    }
    mock_choice = ChatCompletionChoice(index=0, message=Message(role="assistant", content=json.dumps(mock_llm_output)), finish_reason="stop")
    mock_usage = ChatCompletionUsage(prompt_tokens=50, completion_tokens=150, total_tokens=200)
    mock_llm_response = ChatCompletionResponse(id="qa_resp_succ", choices=[mock_choice], model="gpt-test", created=123, usage=mock_usage, provider="test")
    qa_agent._use_llm.return_value = mock_llm_response

    result = await qa_agent.execute_task(task, context)

    qa_agent._use_llm.assert_called_once()
    # Optional: More detailed check of the prompt passed to _use_llm
    prompt_arg = qa_agent._use_llm.call_args[0][0][0].content # Get content of the first message
    assert content_to_review in prompt_arg
    assert criteria in prompt_arg

    assert result.status == TaskStatus.COMPLETED
    assert result.output == mock_llm_output
    assert result.tokens_used == 200
    assert result.error_message is None

@pytest.mark.asyncio
async def test_qa_agent_fallback_to_description_as_content(qa_agent, qa_agent_config):
    # Test when raw_instruction is None, task.description becomes content_to_qa
    task_description_content = "This is the content directly from task description for QA."
    task = Task(
        task_id="qa_task_desc_content",
        description=task_description_content,
        raw_instruction=None # No raw_instruction
    )
    context = TaskContext(current_task=task)

    mock_llm_output = {"overall_assessment": "Approved", "issues_found": [], "suggestions_for_improvement": [], "confidence_score": 0.9}
    mock_choice = ChatCompletionChoice(index=0, message=Message(role="assistant", content=json.dumps(mock_llm_output)), finish_reason="stop")
    mock_usage = ChatCompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    mock_llm_response = ChatCompletionResponse(id="qa_resp_desc", choices=[mock_choice], model="gpt-test", created=123, usage=mock_usage, provider="test")
    qa_agent._use_llm.return_value = mock_llm_response

    result = await qa_agent.execute_task(task, context)

    qa_agent._use_llm.assert_called_once()
    prompt_arg = qa_agent._use_llm.call_args[0][0][0].content
    assert task_description_content in prompt_arg # Check description was used
    assert "General quality check" in prompt_arg # Default criteria

    assert result.status == TaskStatus.COMPLETED
    assert result.output == mock_llm_output
    assert result.tokens_used == 30

@pytest.mark.asyncio
async def test_qa_agent_missing_content_to_qa(qa_agent, qa_agent_config):
    # raw_instruction is JSON but missing 'content_to_qa', and description is also empty (edge case)
    task = Task(
        task_id="qa_task_no_content",
        description="", # Empty description
        raw_instruction=json.dumps({"qa_criteria": "Check something."}) # No content_to_qa
    )
    context = TaskContext(current_task=task)

    result = await qa_agent.execute_task(task, context)

    qa_agent._use_llm.assert_not_called()
    assert result.status == TaskStatus.FAILED
    assert "No 'content_to_qa' provided or determined" in result.error_message

@pytest.mark.asyncio
async def test_qa_agent_llm_empty_response(qa_agent, qa_agent_config):
    raw_instruction_dict = {"content_to_qa": "Test content", "qa_criteria": "Test criteria"}
    task = Task(task_id="qa_task_empty_llm", description="QA", raw_instruction=json.dumps(raw_instruction_dict))
    context = TaskContext(current_task=task)

    # Simulate LLM returning no choices or empty message content
    mock_empty_response = ChatCompletionResponse(id="qa_resp_empty", choices=[], model="gpt-test", created=123, usage=None, provider="test")
    qa_agent._use_llm.return_value = mock_empty_response

    result = await qa_agent.execute_task(task, context)

    assert result.status == TaskStatus.FAILED
    assert "LLM response was empty or malformed for QA" in result.error_message

@pytest.mark.asyncio
async def test_qa_agent_llm_output_not_json(qa_agent, qa_agent_config):
    raw_instruction_dict = {"content_to_qa": "Test content", "qa_criteria": "Test criteria"}
    task = Task(task_id="qa_task_not_json", description="QA", raw_instruction=json.dumps(raw_instruction_dict))
    context = TaskContext(current_task=task)

    non_json_string = "This is not a JSON object."
    mock_choice = ChatCompletionChoice(index=0, message=Message(role="assistant", content=non_json_string), finish_reason="stop")
    mock_llm_response = ChatCompletionResponse(id="qa_resp_notjson", choices=[mock_choice], model="gpt-test",created=123, usage=None, provider="test")
    qa_agent._use_llm.return_value = mock_llm_response

    result = await qa_agent.execute_task(task, context)

    assert result.status == TaskStatus.FAILED
    assert "Failed to parse LLM JSON response for QA" in result.error_message
    assert non_json_string in result.error_message # Check if part of the raw response is included

@pytest.mark.asyncio
async def test_qa_agent_llm_call_exception(qa_agent, qa_agent_config):
    raw_instruction_dict = {"content_to_qa": "Test content", "qa_criteria": "Test criteria"}
    task = Task(task_id="qa_task_exc", description="QA", raw_instruction=json.dumps(raw_instruction_dict))
    context = TaskContext(current_task=task)

    qa_agent._use_llm.side_effect = Exception("LLM API is down")

    result = await qa_agent.execute_task(task, context)

    assert result.status == TaskStatus.FAILED
    assert "Unexpected error in QAAgent: LLM API is down" in result.error_message
