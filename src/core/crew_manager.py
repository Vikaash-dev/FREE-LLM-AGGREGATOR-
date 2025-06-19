from typing import List, Dict, Optional, Any, Type

import structlog

from src.core.agent_structures import AgentConfig, RoleDefinition, TaskContext, TaskResult # Added TaskContext, TaskResult
from src.core.base_agent import AbstractBaseAgent
from src.core.aggregator import LLMAggregator
from src.core.tools_registry import ToolsRegistry
from src.core.planning_structures import ExecutionPlan, ProjectContext, Task, TaskStatus # Added Task, TaskStatus

# Import concrete agent classes
from src.core.agents.planner_agent import PlannerAgent
from src.core.agents.developer_agent import DeveloperAgent
from src.core.agents.researcher_agent import ResearcherAgent
from src.core.agents.qa_agent import QAAgent # Added QAAgent import

logger = structlog.get_logger(__name__)

# Define a mapping from role names (RoleDefinition.name) to agent classes
AGENT_CLASS_MAP: Dict[str, Type[AbstractBaseAgent]] = {
    "SoftwarePlanner": PlannerAgent,
    "PythonDeveloper": DeveloperAgent,
    "GenericResearcher": ResearcherAgent,
    "QualityAssuranceAgent": QAAgent, # Added QAAgent to the map
    # These names MUST match the `name` field in the RoleDefinition objects
    # that will be passed to the CrewManager.
}


class CrewManager:
    def __init__(
        self,
        agent_configs: List[AgentConfig],
        llm_aggregator: LLMAggregator,
        tools_registry: ToolsRegistry,
        role_definitions: Dict[str, RoleDefinition], # Keyed by RoleDefinition.name
    ):
        self.agent_configs = agent_configs
        self.llm_aggregator = llm_aggregator
        self.tools_registry = tools_registry
        self.role_definitions = role_definitions # Store role definitions
        self.agents: Dict[str, AbstractBaseAgent] = {} # Stores agent_id -> agent_instance
        self.agents_by_role: Dict[str, List[AbstractBaseAgent]] = {} # Stores role_name -> list of agent_instances

        logger.info("CrewManager initializing...")
        self._create_agents()

    def _create_agents(self):
        """
        Instantiates agents based on their configurations and assigned roles.
        """
        logger.info("Creating agents based on configurations...")
        for config in self.agent_configs:
            role_name_for_agent = config.role_name # This refers to RoleDefinition.name

            if role_name_for_agent not in self.role_definitions:
                logger.error(f"RoleDefinition for role '{role_name_for_agent}' not found. Cannot create agent {config.agent_id}.")
                continue

            # RoleDefinition object itself is not directly used here for class mapping,
            # but it's good practice to ensure it exists.
            # role_def_obj = self.role_definitions[role_name_for_agent]

            agent_class = AGENT_CLASS_MAP.get(role_name_for_agent)

            if agent_class:
                try:
                    agent_instance = agent_class(
                        agent_config=config,
                        llm_aggregator=self.llm_aggregator,
                        tools_registry=self.tools_registry
                    )
                    self.agents[config.agent_id] = agent_instance

                    if role_name_for_agent not in self.agents_by_role:
                        self.agents_by_role[role_name_for_agent] = []
                    self.agents_by_role[role_name_for_agent].append(agent_instance)

                    logger.info(f"Agent {config.agent_id} of role '{role_name_for_agent}' (class {agent_class.__name__}) created and registered.")
                except Exception as e:
                    logger.error(f"Failed to instantiate agent {config.agent_id} for role {role_name_for_agent}", error=str(e), exc_info=True)
            else:
                logger.warn(f"No agent class found mapped for role_name '{role_name_for_agent}'. Agent {config.agent_id} not created.")
        logger.info(f"Agent creation complete. Total agents created: {len(self.agents)}")


    def _get_agent_for_task(self, task: Task) -> Optional[AbstractBaseAgent]:
        """
        Determines which agent should execute a given task based on task.required_role.
        """
        if not task.required_role:
            logger.warn(f"Task '{task.task_id}' has no required_role specified. Cannot assign agent.", task_description=task.description)
            return None

        agents_for_role = self.agents_by_role.get(task.required_role)
        if not agents_for_role:
            logger.warn(f"No agents found for required_role '{task.required_role}' for task '{task.task_id}'.", task_description=task.description)
            return None

        # Return the first available agent for that role
        # Future: Implement round-robin or load balancing if multiple agents share a role.
        selected_agent = agents_for_role[0]
        logger.info(f"Assigning task '{task.task_id}' to agent '{selected_agent.agent_config.agent_id}' for role '{task.required_role}'.")
        return selected_agent


    async def process_execution_plan(
        self, plan: ExecutionPlan, project_context: Optional[ProjectContext]
    ) -> ExecutionPlan:
        logger.info(f"Processing execution plan: {plan.plan_id}, User Instruction: '{plan.user_instruction[:50]}...'")

        if not self.agents:
            logger.error("No agents available in CrewManager. Cannot process execution plan.", plan_id=plan.plan_id)
            plan.overall_status = TaskStatus.FAILED
            return plan

        # Basic sequential processing for now.
        for current_task_obj in plan.tasks:
            if current_task_obj.status == TaskStatus.COMPLETED:
                logger.info(f"Task {current_task_obj.task_id} is already completed. Skipping.", task_id=current_task_obj.task_id)
                continue

            logger.info(f"Processing task: {current_task_obj.task_id} ('{current_task_obj.description[:50]}...') with required role: {current_task_obj.required_role}", task_id=current_task_obj.task_id)
            current_task_obj.status = TaskStatus.IN_PROGRESS

            agent_to_execute = self._get_agent_for_task(current_task_obj)

            if agent_to_execute:
                task_context = TaskContext(
                    project_context=project_context,
                    current_task=current_task_obj,
                    execution_plan=plan,
                )

                logger.info(f"Executing task {current_task_obj.task_id} with agent {agent_to_execute.agent_config.agent_id} (Role: {agent_to_execute.agent_config.role_name})", task_id=current_task_obj.task_id)

                try:
                    task_result_agent: TaskResult = await agent_to_execute.execute_task(current_task_obj, task_context)

                    current_task_obj.status = task_result_agent.status
                    current_task_obj.output = task_result_agent.output # Ensure output is updated

                    error_msg_detail = ""
                    if task_result_agent.error_message:
                        error_msg_detail = f" Agent error: {task_result_agent.error_message}"
                        current_task_obj.reasoning_log.append(f"Error: {task_result_agent.error_message}")

                    logger.info(f"Task {current_task_obj.task_id} completed with status {task_result_agent.status}.{error_msg_detail}", task_id=current_task_obj.task_id)

                    if task_result_agent.status == TaskStatus.FAILED:
                        logger.error(f"Task {current_task_obj.task_id} failed. Stopping plan processing.{error_msg_detail}", task_id=current_task_obj.task_id)
                        plan.overall_status = TaskStatus.FAILED
                        # Add error message to task if not already set by agent
                        if not hasattr(current_task_obj, 'error_message') or not current_task_obj.error_message: # type: ignore
                             current_task_obj.error_message = f"Task failed with agent {agent_to_execute.agent_config.agent_id}.{error_msg_detail}" # type: ignore
                        return plan

                except Exception as e:
                    logger.error(f"An unexpected error occurred while agent {agent_to_execute.agent_config.agent_id} was executing task {current_task_obj.task_id}", error=str(e), exc_info=True, task_id=current_task_obj.task_id)
                    current_task_obj.status = TaskStatus.FAILED
                    current_task_obj.output = None
                    current_task_obj.reasoning_log.append(f"Critical Error: {str(e)}")
                    current_task_obj.error_message = f"Critical error during task execution: {str(e)}" # type: ignore
                    plan.overall_status = TaskStatus.FAILED
                    return plan

            else:
                # No agent found for the required role or role not specified
                error_reason = f"No agent found for required role '{current_task_obj.required_role}'" if current_task_obj.required_role else "No required_role specified"
                task_error_message = f"{error_reason} for task {current_task_obj.task_id} ('{current_task_obj.description[:50]}...')."

                logger.error(task_error_message, task_id=current_task_obj.task_id)
                current_task_obj.status = TaskStatus.FAILED
                current_task_obj.reasoning_log.append(task_error_message)
                current_task_obj.error_message = task_error_message # type: ignore
                plan.overall_status = TaskStatus.FAILED
                return plan # Stop processing if no agent can handle a task

        # If all tasks completed successfully
        if all(t.status == TaskStatus.COMPLETED for t in plan.tasks):
            plan.overall_status = TaskStatus.COMPLETED
            logger.info(f"Execution plan {plan.plan_id} completed successfully.", plan_id=plan.plan_id)
        elif any(t.status == TaskStatus.FAILED for t in plan.tasks):
             plan.overall_status = TaskStatus.FAILED # Should have been set earlier, but as a safeguard
             logger.warn(f"Execution plan {plan.plan_id} marked as FAILED due to one or more failed tasks.", plan_id=plan.plan_id)
        else:
            # This case (e.g. some pending, some in_progress but loop finished) might indicate an issue
            # or that the plan is not fully terminal. For now, set to in_progress if not all completed.
            plan.overall_status = TaskStatus.IN_PROGRESS
            logger.info(f"Execution plan {plan.plan_id} processing finished, but not all tasks are COMPLETED. Current status: {plan.overall_status}", plan_id=plan.plan_id)

        return plan


# Ensure type hints are resolvable if classes are not yet fully imported elsewhere
if "AgentConfig" not in globals(): # agent_structures.AgentConfig
    class AgentConfig: role_name: str; agent_id: str
if "RoleDefinition" not in globals(): # agent_structures.RoleDefinition
    class RoleDefinition: name: str
if "TaskContext" not in globals(): # agent_structures.TaskContext
    class TaskContext: pass
if "TaskResult" not in globals(): # agent_structures.TaskResult
    class TaskResult: status: TaskStatus; output: Any; error_message: Optional[str]
if "AbstractBaseAgent" not in globals(): # base_agent.AbstractBaseAgent
    class AbstractBaseAgent: agent_config: AgentConfig; async def execute_task(self, t: Task, tc: TaskContext) -> TaskResult: pass
if "LLMAggregator" not in globals(): # core.aggregator.LLMAggregator
    class LLMAggregator: pass
if "ToolsRegistry" not in globals(): # core.tools_registry.ToolsRegistry
    class ToolsRegistry: pass
if "ExecutionPlan" not in globals(): # planning_structures.ExecutionPlan
    class ExecutionPlan: plan_id: str; user_instruction: str; tasks: List[Task]; overall_status: TaskStatus
if "ProjectContext" not in globals(): # planning_structures.ProjectContext
    class ProjectContext: pass
if "Task" not in globals(): # planning_structures.Task
    class Task: task_id: str; description: str; status: TaskStatus; dependencies: List[str]; output: Any; reasoning_log: List[str]
if "TaskStatus" not in globals(): # planning_structures.TaskStatus
    class TaskStatus: PENDING="pending"; IN_PROGRESS="in_progress"; COMPLETED="completed"; FAILED="failed"

# To satisfy type hints for AGENT_CLASS_MAP values
if "PlannerAgent" not in globals():
    class PlannerAgent(AbstractBaseAgent): pass # type: ignore
if "DeveloperAgent" not in globals():
    class DeveloperAgent(AbstractBaseAgent): pass # type: ignore
if "ResearcherAgent" not in globals():
    class ResearcherAgent(AbstractBaseAgent): pass # type: ignore
if "QAAgent" not in globals(): # Added for QAAgent
    class QAAgent(AbstractBaseAgent): pass # type: ignore
