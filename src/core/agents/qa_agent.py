import json
from typing import Optional, Dict, List, Any, cast # Added cast

import structlog

from src.core.base_agent import AbstractBaseAgent, LLMAggregator, ToolsRegistry
from src.core.agent_structures import AgentConfig, TaskContext, TaskResult
# Task is planning_structures.Task, TaskStatus is planning_structures.TaskStatus
from src.core.planning_structures import Task, TaskStatus
from src.models import ChatMessage, ChatCompletionResponse # For LLM interaction

logger = structlog.get_logger(__name__)

class QAAgent(AbstractBaseAgent):
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_aggregator: LLMAggregator,
        tools_registry: Optional[ToolsRegistry] = None,
    ):
        super().__init__(agent_config, llm_aggregator, tools_registry)
        logger.info(f"QAAgent ({self.agent_config.agent_id}) for role '{self.agent_config.role_name}' initialized.")

    async def execute_task(self, task: Task, context: TaskContext) -> TaskResult:
        # task here is planning_structures.Task, because context.current_task is planning_structures.Task
        current_task_obj = context.current_task
        task_id_to_report = current_task_obj.task_id

        logger.info(f"QAAgent '{self.agent_config.agent_id}' executing QA task: {task_id_to_report} - {current_task_obj.description}")

        content_to_qa: Optional[Any] = None
        qa_criteria: str = "General quality check, correctness, and adherence to implicit requirements."

        # The task object from planning_structures.Task does not have a 'details' field.
        # We need to decide how content_to_qa and qa_criteria are passed.
        # Option 1: Use task.description itself if it's simple QA.
        # Option 2: Expect them in task.output from a previous step (if this QA agent is part of a chain).
        # Option 3: For now, let's assume they might be in task.raw_instruction if it's structured,
        # or we could parse from task.description.
        # For this implementation, let's assume they are passed via a dictionary in task.raw_instruction
        # or fallback to using task.description as content_to_qa.

        # Attempt to parse from raw_instruction if it's a JSON string with expected keys
        if current_task_obj.raw_instruction:
            try:
                details_dict = json.loads(current_task_obj.raw_instruction)
                if isinstance(details_dict, dict):
                    content_to_qa = details_dict.get("content_to_qa")
                    qa_criteria = details_dict.get("qa_criteria", qa_criteria)
                    logger.info("Successfully parsed 'content_to_qa' and 'qa_criteria' from task.raw_instruction.")
            except json.JSONDecodeError:
                logger.warn("task.raw_instruction is not a valid JSON string for QA details. Falling back to task.description for content_to_qa.", raw_instruction=current_task_obj.raw_instruction)
                content_to_qa = current_task_obj.description # Fallback
        else:
             # If no raw_instruction, use description as content_to_qa.
             # This implies the task description itself is the content to be QA'd.
            content_to_qa = current_task_obj.description
            logger.info("Using task.description as content_to_qa as task.raw_instruction is empty.")


        if not content_to_qa:
            logger.error("No 'content_to_qa' could be determined for the QA task.")
            return TaskResult(
                task_id=task_id_to_report,
                status=TaskStatus.FAILED,
                error_message="No 'content_to_qa' provided or determined from task details/description."
            )

        # Ensure content_to_qa is string for the prompt
        content_to_qa_str = str(content_to_qa)

        prompt = f"""You are a Quality Assurance Agent. Your task is to review the provided content based on the given criteria and output your findings in a structured JSON format.

Content to QA:
---
{content_to_qa_str}
---

QA Criteria:
---
{qa_criteria}
---

Please provide your QA assessment in the following JSON format:
{{
  "overall_assessment": "string (e.g., 'Approved', 'Approved with minor revisions', 'Requires major revisions', 'Rejected')",
  "issues_found": [
    {{
      "issue_id": "string (e.g., 'ISSUE-001')",
      "description": "string (detailed description of the issue)",
      "severity": "string (e.g., 'Critical', 'High', 'Medium', 'Low')",
      "location_suggestion": "string (optional, e.g., 'Section 2, Paragraph 3' or 'Function X, line Y')",
      "suggested_fix": "string (optional, specific suggestion for fixing)"
    }}
  ],
  "suggestions_for_improvement": [
    "string (general suggestions not tied to specific issues)"
  ],
  "confidence_score": "float (0.0 to 1.0, your confidence in this assessment)"
}}

Output ONLY the JSON object.
"""
        try:
            logger.debug("Sending QA request to LLM.")
            llm_response_obj: ChatCompletionResponse = await self._use_llm([ChatMessage(role="user", content=prompt)])

            if not llm_response_obj.choices or \
               not llm_response_obj.choices[0].message or \
               not llm_response_obj.choices[0].message.content:
                logger.error("LLM response was empty or malformed for QA.")
                return TaskResult(
                    task_id=task_id_to_report,
                    status=TaskStatus.FAILED,
                    error_message="LLM response was empty or malformed for QA."
                )

            response_content = llm_response_obj.choices[0].message.content.strip()

            # Attempt to clean potential markdown fences if LLM adds them
            if response_content.startswith("```json"):
                response_content = response_content[len("```json"):]
            if response_content.endswith("```"):
                response_content = response_content[:-len("```")]
            response_content = response_content.strip()

            try:
                qa_output = json.loads(response_content)
                logger.info("Successfully parsed LLM JSON response for QA.")
                tokens_used = llm_response_obj.usage.total_tokens if llm_response_obj.usage else None
                return TaskResult(
                    task_id=task_id_to_report,
                    status=TaskStatus.COMPLETED,
                    output=qa_output,
                    tokens_used=tokens_used
                )
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse LLM JSON response for QA: {e}. Response (first 200 chars): {response_content[:200]}"
                logger.error(error_msg, raw_llm_response=response_content)
                return TaskResult(
                    task_id=task_id_to_report,
                    status=TaskStatus.FAILED,
                    error_message=error_msg
                )

        except Exception as e:
            logger.error(f"An unexpected error occurred during QA task execution for task {task_id_to_report}: {e}", exc_info=True)
            return TaskResult(
                task_id=task_id_to_report,
                status=TaskStatus.FAILED,
                error_message=f"Unexpected error in QAAgent: {str(e)}"
            )

# Type hinting guards for development if imports are tricky
if "AgentConfig" not in globals():
    class AgentConfig: agent_id: str; role_name: str; llm_config: Optional[Dict]
if "TaskContext" not in globals():
    class TaskContext: current_task: Any; project_context: Any
if "TaskStatus" not in globals():
    class TaskStatus: COMPLETED="COMPLETED"; FAILED="FAILED"
if "Task" not in globals(): # planning_structures.Task
    # Add 'details' for this agent's expectation, though it's not on the formal model yet.
    # This will be handled by how tasks are constructed for this agent.
    class Task: description: str; task_id: str; raw_instruction: Optional[str]; details: Optional[Dict[str, Any]]
if "AbstractBaseAgent" not in globals():
    class AbstractBaseAgent: agent_config: AgentConfig; llm_aggregator: Any; tools_registry: Optional[ToolsRegistry]; def __init__(self,ac, llma, tr): pass; async def _use_llm(self, m) -> Any: pass
if "LLMAggregator" not in globals():
    class LLMAggregator: pass
if "ToolsRegistry" not in globals():
    class ToolsRegistry: pass
if "ChatMessage" not in globals():
    class ChatMessage: pass
if "ChatCompletionResponse" not in globals():
    class ChatCompletionResponse: pass
