import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.agents.planner_agent import PlannerAgent
from src.core.agent_structures import AgentConfig, TaskContext, TaskResult
from src.core.planning_structures import Task, TaskStatus, ExecutionPlan, ProjectContext
from src.core.aggregator import LLMAggregator
from src.core.planner import DevikaInspiredPlanner # To mock its methods

@pytest.fixture
def mock_llm_aggregator_for_planner():
    mock = MagicMock(spec=LLMAggregator)
    mock.chat_completion = AsyncMock() # Planner uses it via DevikaInspiredPlanner
    return mock

@pytest.fixture
def mock_tools_registry_for_planner():
    return MagicMock() # PlannerAgent doesn't directly use tools_registry in current form

@pytest.fixture
def planner_agent_config():
    return AgentConfig(agent_id="planner_001", role_name="SoftwarePlanner")

@pytest.fixture
def sample_task_for_planner():
    # This Task is from planning_structures
    return Task(
        task_id="task_plan_1",
        description="Plan the development of a new web application.",
        raw_instruction="User wants a new web app for task management."
    )

@pytest.fixture
def sample_task_context_for_planner(sample_task_for_planner):
    # TaskContext is from agent_structures
    # It expects current_task to be planning_structures.Task
    project_ctx = ProjectContext(project_id="proj_planner_test", project_name="WebApp Project")
    return TaskContext(
        current_task=sample_task_for_planner,
        project_context=project_ctx
    )

# We need to mock DevikaInspiredPlanner which is instantiated inside PlannerAgent
# We can patch the class itself in the module where PlannerAgent imports it.
@patch('src.core.agents.planner_agent.DevikaInspiredPlanner', autospec=True)
@pytest.mark.asyncio
async def test_planner_agent_execute_task_success(
    MockDevikaInspiredPlanner, # This is the patched class
    planner_agent_config,
    mock_llm_aggregator_for_planner, # LLMAggregator is passed to PlannerAgent, then to DevikaInspiredPlanner
    mock_tools_registry_for_planner,
    sample_task_for_planner, # This is planning_structures.Task
    sample_task_context_for_planner # This is agent_structures.TaskContext
):
    # Configure the mock instance that DevikaInspiredPlanner() will produce
    mock_planner_instance = MockDevikaInspiredPlanner.return_value
    mock_planner_instance.parse_user_intent = AsyncMock(
        return_value={"goal": "parsed_goal", "raw_instruction": sample_task_for_planner.raw_instruction}
    )

    mock_execution_plan = ExecutionPlan(
        plan_id="plan_output_123",
        user_instruction=sample_task_for_planner.raw_instruction, # type: ignore
        tasks=[Task(description="Sub-task 1")] # Add a sub-task for realism
    )
    mock_planner_instance.decompose_complex_task = AsyncMock(return_value=mock_execution_plan)

    # Instantiate PlannerAgent. This will use the mocked DevikaInspiredPlanner.
    agent = PlannerAgent(
        agent_config=planner_agent_config,
        llm_aggregator=mock_llm_aggregator_for_planner, # This will be passed to mocked DevikaInspiredPlanner
        tools_registry=mock_tools_registry_for_planner
    )

    # The 'task' argument to execute_task is formally planning_structures.Task,
    # but the agent primarily uses context.current_task.
    # Here, sample_task_for_planner is used as context.current_task.
    result = await agent.execute_task(sample_task_for_planner, sample_task_context_for_planner)

    # Assert DevikaInspiredPlanner methods were called correctly
    mock_planner_instance.parse_user_intent.assert_called_once_with(
        instruction=sample_task_for_planner.raw_instruction, # Or description if raw_instruction is None
        context=sample_task_context_for_planner.project_context
    )
    mock_planner_instance.decompose_complex_task.assert_called_once_with(
        parsed_intent={"goal": "parsed_goal", "raw_instruction": sample_task_for_planner.raw_instruction},
        context=sample_task_context_for_planner.project_context
    )

    # Assert the TaskResult
    assert isinstance(result, TaskResult)
    assert result.task_id == sample_task_for_planner.task_id
    assert result.status == TaskStatus.COMPLETED
    assert result.output == mock_execution_plan
    assert result.error_message is None


@patch('src.core.agents.planner_agent.DevikaInspiredPlanner', autospec=True)
@pytest.mark.asyncio
async def test_planner_agent_execute_task_planning_failure(
    MockDevikaInspiredPlanner,
    planner_agent_config,
    mock_llm_aggregator_for_planner,
    mock_tools_registry_for_planner,
    sample_task_for_planner,
    sample_task_context_for_planner
):
    mock_planner_instance = MockDevikaInspiredPlanner.return_value
    mock_planner_instance.parse_user_intent = AsyncMock(
        return_value={"goal": "parsed_goal", "raw_instruction": sample_task_for_planner.raw_instruction}
    )
    # Simulate failure in decomposition
    error_message = "Failed to decompose task due to LLM error."
    mock_planner_instance.decompose_complex_task = AsyncMock(side_effect=Exception(error_message))

    agent = PlannerAgent(
        agent_config=planner_agent_config,
        llm_aggregator=mock_llm_aggregator_for_planner,
        tools_registry=mock_tools_registry_for_planner
    )

    result = await agent.execute_task(sample_task_for_planner, sample_task_context_for_planner)

    assert isinstance(result, TaskResult)
    assert result.task_id == sample_task_for_planner.task_id
    assert result.status == TaskStatus.FAILED
    assert result.output is None
    assert error_message in result.error_message # type: ignore

@patch('src.core.agents.planner_agent.DevikaInspiredPlanner', autospec=True)
@pytest.mark.asyncio
async def test_planner_agent_task_with_no_description(
    MockDevikaInspiredPlanner,
    planner_agent_config,
    mock_llm_aggregator_for_planner,
    mock_tools_registry_for_planner,
    sample_task_context_for_planner # Original context
):
    # Modify the task in context to have no description or raw_instruction
    task_no_desc = Task(task_id="task_no_desc_1", description="", raw_instruction=None)
    context_no_desc = TaskContext(
        current_task=task_no_desc,
        project_context=sample_task_context_for_planner.project_context
    )

    mock_planner_instance = MockDevikaInspiredPlanner.return_value # Get the instance

    agent = PlannerAgent(
        agent_config=planner_agent_config,
        llm_aggregator=mock_llm_aggregator_for_planner,
        tools_registry=mock_tools_registry_for_planner
    )

    result = await agent.execute_task(task_no_desc, context_no_desc)

    assert isinstance(result, TaskResult)
    assert result.task_id == task_no_desc.task_id
    assert result.status == TaskStatus.FAILED
    assert "Task has no description or raw instruction" in result.error_message # type: ignore
    mock_planner_instance.parse_user_intent.assert_not_called()
    mock_planner_instance.decompose_complex_task.assert_not_called()
