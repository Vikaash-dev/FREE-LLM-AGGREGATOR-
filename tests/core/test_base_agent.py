import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.base_agent import AbstractBaseAgent
from src.core.agent_structures import AgentConfig, TaskContext, TaskResult
from src.core.planning_structures import Task, TaskStatus # Actual Task and TaskStatus
from src.core.aggregator import LLMAggregator
from src.models import ChatMessage, ChatCompletionRequest, ChatCompletionResponse, Choice, Message

# Define a concrete subclass for testing AbstractBaseAgent
class ConcreteTestAgent(AbstractBaseAgent):
    async def execute_task(self, task: Task, context: TaskContext) -> TaskResult:
        # This method is abstract in AbstractBaseAgent, so it needs to be implemented
        # For testing _use_llm, its actual behavior might not matter much.
        return TaskResult(task_id=task.task_id, status=TaskStatus.COMPLETED, output="test_output")

@pytest.fixture
def mock_llm_aggregator():
    mock = MagicMock(spec=LLMAggregator)
    # Ensure chat_completion is an AsyncMock if it's an async method
    mock.chat_completion = AsyncMock()
    return mock

@pytest.fixture
def agent_config_minimal():
    return AgentConfig(agent_id="test_agent_01", role_name="TestRole")

@pytest.fixture
def concrete_agent(agent_config_minimal, mock_llm_aggregator):
    return ConcreteTestAgent(
        agent_config=agent_config_minimal,
        llm_aggregator=mock_llm_aggregator
        # tools_registry can be None by default
    )

def test_abstract_base_agent_instantiation(concrete_agent, agent_config_minimal, mock_llm_aggregator):
    """Test that AbstractBaseAgent (via ConcreteTestAgent) stores attributes correctly."""
    assert concrete_agent.agent_config == agent_config_minimal
    assert concrete_agent.llm_aggregator == mock_llm_aggregator
    assert concrete_agent.tools_registry is None

@pytest.mark.asyncio
async def test_use_llm_default_config(concrete_agent, mock_llm_aggregator):
    """Test _use_llm with default agent LLM config (which is None here)."""
    messages = [ChatMessage(role="user", content="Hello")]

    # Mock the response from llm_aggregator.chat_completion
    mock_response_content = "LLM says hi"
    mock_chat_completion_response = ChatCompletionResponse(
        id="chatcmpl-test",
        object="chat.completion",
        created=1234567890,
        model="gpt-test",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=mock_response_content),
                finish_reason="stop",
            )
        ],
    )
    mock_llm_aggregator.chat_completion.return_value = mock_chat_completion_response

    response = await concrete_agent._use_llm(messages)

    assert response == mock_chat_completion_response
    # Check that chat_completion was called with the correct ChatCompletionRequest
    mock_llm_aggregator.chat_completion.assert_called_once()
    call_args = mock_llm_aggregator.chat_completion.call_args[0][0] # Get the first positional argument
    assert isinstance(call_args, ChatCompletionRequest)
    assert call_args.messages == messages
    assert call_args.model == "auto" # Default when no config is set in AgentConfig or specific_llm_config

@pytest.mark.asyncio
async def test_use_llm_with_agent_config(mock_llm_aggregator):
    """Test _use_llm when AgentConfig has llm_config."""
    agent_llm_config = {"model": "gpt-3.5-turbo", "temperature": 0.5, "max_tokens": 100}
    config = AgentConfig(
        agent_id="test_agent_02",
        role_name="ConfiguredRole",
        llm_config=agent_llm_config
    )
    agent = ConcreteTestAgent(agent_config=config, llm_aggregator=mock_llm_aggregator)

    messages = [ChatMessage(role="user", content="Test with agent config")]
    mock_llm_aggregator.chat_completion.return_value = ChatCompletionResponse(
        id="test", object="chat.completion", created=1, model=agent_llm_config["model"], choices=[]
    ) # Simplified response

    await agent._use_llm(messages)

    mock_llm_aggregator.chat_completion.assert_called_once()
    call_args = mock_llm_aggregator.chat_completion.call_args[0][0]
    assert call_args.model == agent_llm_config["model"]
    assert call_args.temperature == agent_llm_config["temperature"]
    assert call_args.max_tokens == agent_llm_config["max_tokens"]

@pytest.mark.asyncio
async def test_use_llm_with_specific_config(concrete_agent, mock_llm_aggregator):
    """Test _use_llm with specific_llm_config overriding agent's default."""
    messages = [ChatMessage(role="user", content="Hello with specific config")]
    specific_config = {"model": "gpt-4-specific", "temperature": 0.8}

    mock_llm_aggregator.chat_completion.return_value = ChatCompletionResponse(
        id="test_spec", object="chat.completion", created=1, model=specific_config["model"], choices=[]
    ) # Simplified

    await concrete_agent._use_llm(messages, specific_llm_config=specific_config)

    mock_llm_aggregator.chat_completion.assert_called_once()
    call_args = mock_llm_aggregator.chat_completion.call_args[0][0]
    assert call_args.model == specific_config["model"]
    assert call_args.temperature == specific_config["temperature"]
    assert call_args.max_tokens is None # Not in specific_config, and agent_config.llm_config is None

@pytest.mark.asyncio
async def test_use_llm_merging_configs(mock_llm_aggregator):
    """Test _use_llm when specific_llm_config merges with agent_config.llm_config."""
    agent_llm_cfg = {"model": "base-model", "temperature": 0.5, "top_p": 0.9}
    config = AgentConfig(
        agent_id="test_agent_03",
        role_name="MergingRole",
        llm_config=agent_llm_cfg
    )
    agent = ConcreteTestAgent(agent_config=config, llm_aggregator=mock_llm_aggregator)

    messages = [ChatMessage(role="user", content="Test merging")]
    specific_llm_cfg = {"model": "override-model", "max_tokens": 200} # Overrides model, adds max_tokens

    mock_llm_aggregator.chat_completion.return_value = ChatCompletionResponse(
        id="test_merge", object="chat.completion", created=1, model=specific_llm_cfg["model"], choices=[]
    )

    await agent._use_llm(messages, specific_llm_config=specific_llm_cfg)

    mock_llm_aggregator.chat_completion.assert_called_once()
    call_args = mock_llm_aggregator.chat_completion.call_args[0][0]
    assert call_args.model == specific_llm_cfg["model"] # Overridden
    assert call_args.temperature == agent_llm_cfg["temperature"] # From agent_config
    assert call_args.top_p == agent_llm_cfg["top_p"] # From agent_config
    assert call_args.max_tokens == specific_llm_cfg["max_tokens"] # From specific_config
    assert call_args.stream is False # Default from ChatCompletionRequest if not set
