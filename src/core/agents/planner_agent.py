from typing import Any # Added Any for AbstractBaseAgent hint
import structlog

from src.core.base_agent import AbstractBaseAgent, LLMAggregator # Added LLMAggregator
from src.core.agent_structures import AgentConfig, TaskContext, TaskResult # Task removed, AgentConfig added
from src.core.planning_structures import TaskStatus, ExecutionPlan, ProjectContext, Task # Task imported from planning_structures
from src.core.planner import DevikaInspiredPlanner # Import the planner

logger = structlog.get_logger(__name__)


class PlannerAgent(AbstractBaseAgent):
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_aggregator: LLMAggregator,
        tools_registry: Any = None, # tools_registry is Optional[Any] in base
    ):
        super().__init__(agent_config, llm_aggregator, tools_registry)
        self.planner = DevikaInspiredPlanner(llm_aggregator=self.llm_aggregator)
        logger.info(f"PlannerAgent ({self.agent_config.agent_id}) initialized with DevikaInspiredPlanner.")

    async def execute_task(self, task: Task, context: TaskContext) -> TaskResult:
        # Note: task is src.core.planning_structures.Task
        # context.current_task is also src.core.planning_structures.Task
        # This is because TaskContext hints `current_task: Task` which was forward-declared to be planning_structures.Task

        current_task_obj = context.current_task # This is the planning_structures.Task object
        task_id_to_report = current_task_obj.task_id

        logger.info(
            f"PlannerAgent ({self.agent_config.agent_id}) executing task: {task_id_to_report} - {current_task_obj.description}"
        )

        instruction_to_plan = current_task_obj.raw_instruction or current_task_obj.description
        if not instruction_to_plan:
            logger.error(f"Task {task_id_to_report} has no description or raw_instruction.")
            return TaskResult(
                task_id=task_id_to_report,
                status=TaskStatus.FAILED,
                error_message="Task has no description or raw instruction.",
            )

        try:
            logger.info(f"Parsing user intent for instruction: '{instruction_to_plan}'")
            # Pass project_context if available in context
            project_ctx: Optional[ProjectContext] = getattr(context, 'project_context', None)

            parsed_intent = await self.planner.parse_user_intent(
                instruction=instruction_to_plan,
                context=project_ctx
            )
            logger.info(f"Parsed intent: {parsed_intent.get('goal', 'N/A')}")

            logger.info(f"Decomposing complex task based on parsed intent for task: {task_id_to_report}")
            execution_plan_output: ExecutionPlan = await self.planner.decompose_complex_task(
                parsed_intent=parsed_intent,
                context=project_ctx
            )
            logger.info(f"Decomposition complete. Plan ID: {execution_plan_output.plan_id}, Number of sub_tasks: {len(execution_plan_output.tasks)}")

            return TaskResult(
                task_id=task_id_to_report,
                status=TaskStatus.COMPLETED,
                output=execution_plan_output,
                # tokens_used and cost could be aggregated if DevikaInspiredPlanner reported them
            )

        except Exception as e:
            logger.error(
                f"PlannerAgent ({self.agent_config.agent_id}) failed to execute task {task_id_to_report}",
                error=str(e),
                exc_info=True,
            )
            return TaskResult(
                task_id=task_id_to_report,
                status=TaskStatus.FAILED,
                error_message=str(e),
            )

# Type hinting guards for development/static analysis if imports are tricky
# These might need adjustment based on the actual import structure.
# The `Task` used by `execute_task` method will be `planning_structures.Task`
# The `TaskResult` returned must be `agent_structures.TaskResult`
if "AgentConfig" not in globals():
    class AgentConfig: pass
if "TaskContext" not in globals(): # agent_structures.TaskContext
    class TaskContext: current_task: Any; project_context: Any
# if "TaskResult" not in globals(): # agent_structures.TaskResult (already defined in agent_structures)
#     class TaskResult: pass
if "TaskStatus" not in globals(): # planning_structures.TaskStatus
    class TaskStatus: COMPLETED="COMPLETED"; FAILED="FAILED" # Simplified for guard
if "ExecutionPlan" not in globals(): # planning_structures.ExecutionPlan
    class ExecutionPlan: pass
if "ProjectContext" not in globals(): # planning_structures.ProjectContext
    class ProjectContext: pass
if "Task" not in globals(): # planning_structures.Task
    class Task: description: str; task_id: str; raw_instruction: Optional[str]
if "AbstractBaseAgent" not in globals(): # base_agent.AbstractBaseAgent
    class AbstractBaseAgent: agent_config: Any; llm_aggregator: Any; tools_registry: Any; def __init__(self,ac, llma, tr): pass
if "LLMAggregator" not in globals(): # aggregator.LLMAggregator
    class LLMAggregator: pass
