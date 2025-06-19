# Core components
from .agent_structures import RoleDefinition, AgentConfig, TaskContext, TaskResult
from .base_agent import AbstractBaseAgent
from .tools_registry import ToolsRegistry
from .crew_manager import CrewManager
from .tool_interface import ToolInterface, ToolOutput

# Workflow System Structures
from .workflow_structures import WorkflowGraph, WorkflowNode, WorkflowEdge, NodeType
from .workflow_manager import WorkflowManager

# Planning structures (re-exporting for convenience if they are defined elsewhere)
# from .planning_structures import Task, ExecutionPlan, ProjectContext, TaskStatus

# Expose specific agents if needed, or allow them to be imported via their modules
# from .agents.planner_agent import PlannerAgent
# from .agents.developer_agent import DeveloperAgent
# from .agents.researcher_agent import ResearcherAgent

# Expose specific tools or tool collections if needed
# from .tools.file_system_tools import ReadFileTool, WriteFileTool


__all__ = [
    "RoleDefinition",
    "AgentConfig",
    "TaskContext",
    "TaskResult",
    "AbstractBaseAgent",
    "ToolsRegistry",
    "CrewManager",
    "ToolInterface",
    "ToolOutput",
    "WorkflowGraph", # Added
    "WorkflowNode",  # Added
    "WorkflowEdge",  # Added
    "NodeType",      # Added
    "WorkflowManager", # Added
    # "Task", # Uncomment if re-exporting
    # "ExecutionPlan", # Uncomment if re-exporting
    # "ProjectContext", # Uncomment if re-exporting
    # "TaskStatus", # Uncomment if re-exporting
    # "PlannerAgent",
    # "DeveloperAgent",
    # "ResearcherAgent",
    # "ReadFileTool", # Example if tools were directly exposed
    # "WriteFileTool",
]