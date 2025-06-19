import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.core.crew_manager import CrewManager, AGENT_CLASS_MAP
from src.core.agent_structures import AgentConfig, RoleDefinition, TaskContext, TaskResult
from src.core.base_agent import AbstractBaseAgent
from src.core.planning_structures import ExecutionPlan, Task, ProjectContext, TaskStatus
from src.core.aggregator import LLMAggregator
from src.core.tools_registry import ToolsRegistry

from src.core.agents.planner_agent import PlannerAgent
from src.core.agents.developer_agent import DeveloperAgent
from src.core.agents.researcher_agent import ResearcherAgent


@pytest.fixture
def mock_llm_aggregator_for_crew():
    return MagicMock(spec=LLMAggregator)

@pytest.fixture
def mock_tools_registry_for_crew():
    return MagicMock(spec=ToolsRegistry)

@pytest.fixture
def sample_role_definitions():
    return {
        "SoftwarePlanner": RoleDefinition(name="SoftwarePlanner", description="Plans software projects", goals=[], responsibilities=[]),
        "PythonDeveloper": RoleDefinition(name="PythonDeveloper", description="Develops Python code", goals=[], responsibilities=[]),
        "GenericResearcher": RoleDefinition(name="GenericResearcher", description="Conducts research", goals=[], responsibilities=[]),
    }

@pytest.fixture
def sample_agent_configs():
    return [
        AgentConfig(agent_id="planner_1", role_name="SoftwarePlanner"),
        AgentConfig(agent_id="developer_1", role_name="PythonDeveloper"),
        AgentConfig(agent_id="researcher_1", role_name="GenericResearcher"),
    ]

@pytest.fixture
def sample_project_context_for_crew():
    return ProjectContext(project_id="crew_test_project", project_name="Test Project X")

@pytest.fixture
def crew_manager_instance(sample_agent_configs, mock_llm_aggregator_for_crew, mock_tools_registry_for_crew, sample_role_definitions):
    # Helper fixture to get a CrewManager instance
    return CrewManager(
        agent_configs=sample_agent_configs,
        llm_aggregator=mock_llm_aggregator_for_crew,
        tools_registry=mock_tools_registry_for_crew,
        role_definitions=sample_role_definitions
    )


def test_crew_manager_instantiation_and_agent_creation(
    sample_agent_configs, mock_llm_aggregator_for_crew, mock_tools_registry_for_crew, sample_role_definitions
):
    with patch.dict(AGENT_CLASS_MAP, {
        "SoftwarePlanner": MagicMock(spec=PlannerAgent),
        "PythonDeveloper": MagicMock(spec=DeveloperAgent),
        "GenericResearcher": MagicMock(spec=ResearcherAgent),
    }) as mock_agent_map:
        mock_planner_instance = mock_agent_map["SoftwarePlanner"].return_value
        mock_planner_instance.agent_config = sample_agent_configs[0]
        mock_dev_instance = mock_agent_map["PythonDeveloper"].return_value
        mock_dev_instance.agent_config = sample_agent_configs[1]
        mock_researcher_instance = mock_agent_map["GenericResearcher"].return_value
        mock_researcher_instance.agent_config = sample_agent_configs[2]

        crew = CrewManager(sample_agent_configs, mock_llm_aggregator_for_crew, mock_tools_registry_for_crew, sample_role_definitions)

        assert len(crew.agents) == 3
        assert crew.agents["planner_1"] == mock_planner_instance
        assert crew.agents_by_role["SoftwarePlanner"][0] == mock_planner_instance
        mock_agent_map["SoftwarePlanner"].assert_called_once_with(
            agent_config=sample_agent_configs[0], llm_aggregator=mock_llm_aggregator_for_crew, tools_registry=mock_tools_registry_for_crew)
        # ... (similar asserts for other agents)

def test_crew_manager_instantiation_unknown_role(
    sample_agent_configs, mock_llm_aggregator_for_crew, mock_tools_registry_for_crew, sample_role_definitions, caplog
):
    unknown_role_config = AgentConfig(agent_id="unknown_agent", role_name="UnknownRole")
    configs_with_unknown = sample_agent_configs + [unknown_role_config]
    crew = CrewManager(configs_with_unknown, mock_llm_aggregator_for_crew, mock_tools_registry_for_crew, sample_role_definitions)
    assert len(crew.agents) == 3
    assert "unknown_agent" not in crew.agents
    assert "RoleDefinition for role 'UnknownRole' not found" in caplog.text or \
           "No agent class found mapped for role_name 'UnknownRole'" in caplog.text

# --- Tests for _get_agent_for_task ---
def test_get_agent_for_task_success(crew_manager_instance: CrewManager):
    task = Task(task_id="t1", description="Plan task", required_role="SoftwarePlanner")
    agent = crew_manager_instance._get_agent_for_task(task)
    assert agent is not None
    assert agent.agent_config.agent_id == "planner_1"
    assert agent.agent_config.role_name == "SoftwarePlanner"

def test_get_agent_for_task_no_role_specified(crew_manager_instance: CrewManager, caplog):
    task = Task(task_id="t2", description="Task with no role", required_role=None)
    agent = crew_manager_instance._get_agent_for_task(task)
    assert agent is None
    assert "Task 't2' has no required_role specified." in caplog.text

def test_get_agent_for_task_no_agent_for_role(crew_manager_instance: CrewManager, caplog):
    task = Task(task_id="t3", description="Task for unknown role", required_role="NonExistentRole")
    agent = crew_manager_instance._get_agent_for_task(task)
    assert agent is None
    assert "No agents found for required_role 'NonExistentRole' for task 't3'" in caplog.text

# --- Tests for process_execution_plan with new routing ---
@pytest.mark.asyncio
async def test_process_plan_success_new_routing(crew_manager_instance: CrewManager, sample_project_context_for_crew):
    # Mock execute_task for each agent instance
    crew_manager_instance.agents["planner_1"].execute_task = AsyncMock(return_value=TaskResult(task_id="task1", status=TaskStatus.COMPLETED, output="Plan output"))
    crew_manager_instance.agents["developer_1"].execute_task = AsyncMock(return_value=TaskResult(task_id="task2", status=TaskStatus.COMPLETED, output="Code output"))
    crew_manager_instance.agents["researcher_1"].execute_task = AsyncMock(return_value=TaskResult(task_id="task3", status=TaskStatus.COMPLETED, output="Research output"))

    tasks = [
        Task(task_id="task1", description="Plan project", required_role="SoftwarePlanner"),
        Task(task_id="task2", description="Develop module", required_role="PythonDeveloper"),
        Task(task_id="task3", description="Research APIs", required_role="GenericResearcher"),
    ]
    plan = ExecutionPlan(user_instruction="Execute full project", tasks=tasks)
    updated_plan = await crew_manager_instance.process_execution_plan(plan, sample_project_context_for_crew)

    assert updated_plan.overall_status == TaskStatus.COMPLETED
    assert updated_plan.tasks[0].status == TaskStatus.COMPLETED
    assert updated_plan.tasks[0].output == "Plan output"
    crew_manager_instance.agents["planner_1"].execute_task.assert_called_once()
    # ... (similar asserts for other tasks and agent calls) ...

@pytest.mark.asyncio
async def test_process_plan_task_fails_due_to_agent_error(crew_manager_instance: CrewManager, sample_project_context_for_crew):
    crew_manager_instance.agents["planner_1"].execute_task = AsyncMock(return_value=TaskResult(task_id="task1", status=TaskStatus.COMPLETED, output="Plan output"))
    crew_manager_instance.agents["developer_1"].execute_task = AsyncMock(return_value=TaskResult(task_id="task2", status=TaskStatus.FAILED, error_message="Agent dev error"))
    crew_manager_instance.agents["researcher_1"].execute_task = AsyncMock() # Should not be called

    tasks = [
        Task(task_id="task1", description="Plan", required_role="SoftwarePlanner"),
        Task(task_id="task2", description="Develop", required_role="PythonDeveloper"), # This task will fail
        Task(task_id="task3", description="Research", required_role="GenericResearcher"),
    ]
    plan = ExecutionPlan(user_instruction="Test agent failure", tasks=tasks)
    updated_plan = await crew_manager_instance.process_execution_plan(plan, sample_project_context_for_crew)

    assert updated_plan.overall_status == TaskStatus.FAILED
    assert updated_plan.tasks[0].status == TaskStatus.COMPLETED
    assert updated_plan.tasks[1].status == TaskStatus.FAILED
    assert "Agent error: Agent dev error" in updated_plan.tasks[1].reasoning_log[-1] # Check reasoning log for agent error
    assert updated_plan.tasks[2].status == TaskStatus.PENDING # Not processed
    crew_manager_instance.agents["researcher_1"].execute_task.assert_not_called()

@pytest.mark.asyncio
async def test_process_plan_task_fails_no_required_role(crew_manager_instance: CrewManager, sample_project_context_for_crew, caplog):
    tasks = [
        Task(task_id="task_no_role", description="Task missing role", required_role=None)
    ]
    plan = ExecutionPlan(user_instruction="Test missing role", tasks=tasks)
    updated_plan = await crew_manager_instance.process_execution_plan(plan, sample_project_context_for_crew)

    assert updated_plan.overall_status == TaskStatus.FAILED
    assert updated_plan.tasks[0].status == TaskStatus.FAILED
    assert "No required_role specified for task task_no_role" in updated_plan.tasks[0].error_message # type: ignore
    assert "No required_role specified" in caplog.text

@pytest.mark.asyncio
async def test_process_plan_task_fails_no_agent_for_role(crew_manager_instance: CrewManager, sample_project_context_for_crew, caplog):
    tasks = [
        Task(task_id="task_bad_role", description="Task for non-existent role", required_role="ImaginaryRole")
    ]
    plan = ExecutionPlan(user_instruction="Test bad role", tasks=tasks)
    updated_plan = await crew_manager_instance.process_execution_plan(plan, sample_project_context_for_crew)

    assert updated_plan.overall_status == TaskStatus.FAILED
    assert updated_plan.tasks[0].status == TaskStatus.FAILED
    assert "No agent found for required role 'ImaginaryRole' for task task_bad_role" in updated_plan.tasks[0].error_message # type: ignore
    assert "No agents found for required_role 'ImaginaryRole'" in caplog.text

@pytest.mark.asyncio
async def test_crew_manager_no_agents_in_manager(
    mock_llm_aggregator_for_crew, mock_tools_registry_for_crew, sample_role_definitions, sample_project_context_for_crew, caplog
):
    crew = CrewManager([], mock_llm_aggregator_for_crew, mock_tools_registry_for_crew, sample_role_definitions)
    assert len(crew.agents) == 0
    tasks = [Task(task_id="t1", description="A task", required_role="SoftwarePlanner")]
    plan = ExecutionPlan(user_instruction="Test with no agents", tasks=tasks)
    updated_plan = await crew.process_execution_plan(plan, sample_project_context_for_crew)

    assert updated_plan.overall_status == TaskStatus.FAILED
    assert "No agents available in CrewManager" in caplog.text
    assert updated_plan.tasks[0].status == TaskStatus.PENDING # Remains PENDING as processing stops early
