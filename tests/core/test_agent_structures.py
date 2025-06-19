import pytest
from pydantic import ValidationError

from src.core.agent_structures import (
    RoleDefinition,
    AgentConfig,
    TaskContext,
    TaskResult,
)
# Assuming planning_structures types are available for type hinting if needed,
# but for basic instantiation tests, they might not be strictly necessary to import.
# from src.core.planning_structures import Task, ProjectContext, ExecutionPlan, TaskStatus
# For TaskResult, TaskStatus is needed.
from src.core.planning_structures import TaskStatus, Task, ProjectContext, ExecutionPlan

# Minimal Task, ProjectContext, ExecutionPlan for TaskContext testing
class MinimalTask:
    task_id: str = "task_123"
    description: str = "Minimal task"

class MinimalProjectContext:
    project_id: str = "proj_123"

class MinimalExecutionPlan:
    plan_id: str = "plan_123"


def test_role_definition_instantiation():
    """Test basic RoleDefinition instantiation."""
    role_def = RoleDefinition(
        name="TestRole",
        description="A role for testing.",
        goals=["goal1", "goal2"],
        responsibilities=["resp1"],
    )
    assert role_def.name == "TestRole"
    assert role_def.description == "A role for testing."
    assert role_def.goals == ["goal1", "goal2"]
    assert role_def.responsibilities == ["resp1"]
    assert role_def.tools == []  # Default factory
    assert role_def.input_schema is None
    assert role_def.output_schema is None

    role_def_full = RoleDefinition(
        name="TestRoleFull",
        description="A full role for testing.",
        goals=["g1"],
        responsibilities=["r1"],
        tools=["tool1", "tool2"],
        input_schema={"type": "object"},
        output_schema={"type": "string"},
    )
    assert role_def_full.tools == ["tool1", "tool2"]
    assert role_def_full.input_schema == {"type": "object"}
    assert role_def_full.output_schema == {"type": "string"}

def test_role_definition_missing_required_fields():
    """Test RoleDefinition instantiation fails with missing required fields."""
    with pytest.raises(ValidationError):
        RoleDefinition(name="TestRole")  # Missing description, goals, responsibilities

def test_agent_config_instantiation():
    """Test basic AgentConfig instantiation."""
    agent_cfg = AgentConfig(agent_id="agent007", role_name="TestRole")
    assert agent_cfg.agent_id == "agent007"
    assert agent_cfg.role_name == "TestRole"
    assert agent_cfg.llm_config is None
    assert agent_cfg.backstory is None
    assert agent_cfg.allow_delegation is False
    assert agent_cfg.max_iterations == 10 # Default

    agent_cfg_full = AgentConfig(
        agent_id="agent008",
        role_name="SuperRole",
        llm_config={"model": "gpt-4"},
        backstory="Born in a test tube.",
        allow_delegation=True,
        max_iterations=5,
    )
    assert agent_cfg_full.llm_config == {"model": "gpt-4"}
    assert agent_cfg_full.backstory == "Born in a test tube."
    assert agent_cfg_full.allow_delegation is True
    assert agent_cfg_full.max_iterations == 5

def test_task_context_instantiation():
    """Test basic TaskContext instantiation."""
    # Using actual types from planning_structures for current_task
    # but can use MinimalTask if direct import is an issue in some test setups
    # For this test, we'll use the actual Task from planning_structures
    # to ensure compatibility with Pydantic's type validation.

    # First, create a valid Task object from planning_structures
    # (or a compatible mock if direct instantiation is complex)
    current_task_mock = Task(description="Actual task from planning_structures")


    task_ctx = TaskContext(current_task=current_task_mock)
    assert task_ctx.current_task == current_task_mock
    assert task_ctx.project_context is None
    assert task_ctx.execution_plan is None
    assert task_ctx.shared_memory == {}  # Default factory
    assert task_ctx.history == []  # Default factory

    # Create mock/minimal versions for ProjectContext and ExecutionPlan for full test
    project_context_mock = ProjectContext(project_name="Test Project")
    execution_plan_mock = ExecutionPlan(user_instruction="Do something")

    task_ctx_full = TaskContext(
        current_task=current_task_mock,
        project_context=project_context_mock, # type: ignore
        execution_plan=execution_plan_mock, # type: ignore
        shared_memory={"key": "value"},
        history=[{"event": "started"}],
    )
    assert task_ctx_full.project_context == project_context_mock
    assert task_ctx_full.execution_plan == execution_plan_mock
    assert task_ctx_full.shared_memory == {"key": "value"}
    assert task_ctx_full.history == [{"event": "started"}]

def test_task_result_instantiation():
    """Test basic TaskResult instantiation."""
    task_res = TaskResult(task_id="task_001", status=TaskStatus.COMPLETED)
    assert task_res.task_id == "task_001"
    assert task_res.status == TaskStatus.COMPLETED
    assert task_res.output is None
    assert task_res.error_message is None
    assert task_res.tokens_used is None
    assert task_res.cost is None

    task_res_full = TaskResult(
        task_id="task_002",
        status=TaskStatus.FAILED,
        output={"data": "some_output"},
        error_message="Something went wrong.",
        tokens_used=100,
        cost=0.002,
    )
    assert task_res_full.output == {"data": "some_output"}
    assert task_res_full.error_message == "Something went wrong."
    assert task_res_full.tokens_used == 100
    assert task_res_full.cost == 0.002

def test_task_result_invalid_status():
    """Test TaskResult instantiation fails with invalid status type."""
    with pytest.raises(ValidationError):
        TaskResult(task_id="task_003", status="invalid_status_string") # type: ignore
