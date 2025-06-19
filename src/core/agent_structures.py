from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

# Forward declaration for ProjectContext, Task, ExecutionPlan, TaskStatus
# These will be imported from src.core.planning_structures
# This is a common pattern to avoid circular dependencies while allowing type hinting
ProjectContext = "ProjectContext"
Task = "Task"
ExecutionPlan = "ExecutionPlan"
TaskStatus = "TaskStatus"


class RoleDefinition(BaseModel):
    name: str
    description: str
    goals: List[str]
    responsibilities: List[str]
    tools: List[str] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


class AgentConfig(BaseModel):
    agent_id: str
    role_name: str
    llm_config: Optional[Dict[str, Any]] = None
    backstory: Optional[str] = None
    allow_delegation: bool = False
    max_iterations: int = 10


class TaskContext(BaseModel):
    project_context: Optional[ProjectContext] = None
    current_task: Task
    execution_plan: Optional[ExecutionPlan] = None
    shared_memory: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    output: Optional[Any] = None
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
