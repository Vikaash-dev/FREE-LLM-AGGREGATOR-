from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CLARIFICATION_NEEDED = "clarification_needed"
    CANCELLED = "cancelled"

@dataclass
class ProjectContext:
    '''Holds information about the current project or environment.'''
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    project_name: Optional[str] = None
    project_description: Optional[str] = None
    # Example: current working directory, git repo info, etc.
    # For Phase 1, this can be simple.
    additional_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    '''Represents a single task or sub-task within an execution plan.'''
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # List of task_ids this task depends on
    parent_task_id: Optional[str] = None # For hierarchical tasks, if any
    sub_tasks: List['Task'] = field(default_factory=list) # Child tasks for decomposition
    output: Optional[Any] = None # Result or output of the task
    reasoning_log: List[str] = field(default_factory=list) # Log of reasoning steps or decisions for this task
    raw_instruction: Optional[str] = None # The original instruction that led to this task, if applicable

    # To allow Task to have sub_tasks of type Task, we need to handle potential forward references
    # This is generally handled by Python 3.7+ with `from __future__ import annotations`
    # or by using string literals for type hints if issues arise, but dataclasses handle it well.

@dataclass
class ExecutionPlan:
    '''Represents a plan to achieve a high-level goal, composed of multiple tasks.'''
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_instruction: str # The original high-level instruction from the user
    parsed_intent: Dict[str, Any] = field(default_factory=dict) # Structured intent from NLP parsing
    tasks: List[Task] = field(default_factory=list)
    overall_status: TaskStatus = TaskStatus.PENDING
    # Could include estimated time, resources, etc. in future

@dataclass
class TaskResult:
    '''Represents the result of executing a task.'''
    task_id: str
    status: TaskStatus
    message: Optional[str] = None # Any message accompanying the result (e.g., error message)
    output: Optional[Any] = None # Actual output data from the task
