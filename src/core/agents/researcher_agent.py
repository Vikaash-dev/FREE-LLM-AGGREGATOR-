from typing import Any, List, Optional, Dict, Tuple # Added Tuple
import structlog
import re

from src.core.base_agent import AbstractBaseAgent, LLMAggregator, ToolsRegistry
from src.core.agent_structures import AgentConfig, TaskContext, TaskResult
from src.core.planning_structures import TaskStatus, Task, ProjectContext
from src.core.researcher import IntelligentResearchAssistant
from src.core.research_components import ContextualKeywordExtractor, WebResearcher, RelevanceScorer
from src.core.research_structures import ResearchQuery, KnowledgeChunk
from src.core.tool_interface import ToolOutput # For type hinting tool outputs
from src.models import ChatMessage # For _use_llm example

logger = structlog.get_logger(__name__)

def _parse_path_from_description(text: str, keyword: str) -> Optional[str]:
    match = re.search(rf"{keyword}(?:\s+to)?\s*'([^']*)'", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


class ResearcherAgent(AbstractBaseAgent):
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_aggregator: LLMAggregator,
        tools_registry: Optional[ToolsRegistry] = None,
    ):
        super().__init__(agent_config, llm_aggregator, tools_registry)
        # Initialize research components once
        keyword_extractor = ContextualKeywordExtractor(llm_aggregator=self.llm_aggregator)
        web_researcher = WebResearcher()
        relevance_scorer = RelevanceScorer(llm_aggregator=self.llm_aggregator)
        self.research_assistant = IntelligentResearchAssistant(
            keyword_extractor=keyword_extractor,
            web_researcher=web_researcher,
            relevance_scorer=relevance_scorer,
            llm_aggregator=self.llm_aggregator,
        )
        logger.info(f"ResearcherAgent ({self.agent_config.agent_id}) initialized with IntelligentResearchAssistant.")

    async def _sop_clarify_research_goal(self, task_description: str, context: TaskContext) -> str:
        logger.debug(f"Agent {self.agent_config.agent_id}: SOP Step - Clarifying Research Goal for: {task_description[:100]}...")
        # For complex tasks, this might involve an LLM call to refine the research question.
        # Example:
        # messages = [
        #    ChatMessage(role="system", content="You are an expert in formulating precise research questions."),
        #    ChatMessage(role="user", content=f"Based on the task: '{task_description}', what is the core research question or information to find?")
        # ]
        # response = await self._use_llm(messages)
        # clarified_goal = response.choices[0].message.content or task_description
        clarified_goal = task_description # Placeholder for now
        logger.debug(f"Agent {self.agent_config.agent_id}: SOP Step - Research Goal Clarification Complete. Goal: {clarified_goal[:100]}...")
        return clarified_goal

    async def _sop_execute_research_cycle(self, research_goal: str, context: TaskContext) -> Tuple[Optional[List[KnowledgeChunk]], Optional[str]]:
        logger.debug(f"Agent {self.agent_config.agent_id}: SOP Step - Executing Research Cycle for goal: {research_goal[:100]}...")
        error_str: Optional[str] = None
        knowledge_chunks: Optional[List[KnowledgeChunk]] = None

        project_context_summary: Optional[str] = None
        if context.project_context and hasattr(context.project_context, 'project_description'):
            project_context_summary = context.project_context.project_description

        research_query = ResearchQuery(
            original_task_description=research_goal, # Use the (potentially clarified) goal
            keywords=[], # IRA will extract if empty
            project_context_summary=project_context_summary,
        )

        try:
            knowledge_chunks = await self.research_assistant.research_for_task(research_query=research_query)
            logger.info(f"Research cycle found {len(knowledge_chunks) if knowledge_chunks else 0} knowledge chunks.")
            if not knowledge_chunks:
                # This is not strictly an error for the SOP step, but an outcome.
                logger.info("Research cycle completed but yielded no specific knowledge chunks.")
        except Exception as e:
            logger.error(f"Error during research_assistant.research_for_task: {e}", exc_info=True)
            error_str = f"Research execution failed: {str(e)}"
            knowledge_chunks = None # Ensure chunks are None on error

        logger.debug(f"Agent {self.agent_config.agent_id}: SOP Step - Research Cycle Execution Complete.")
        return knowledge_chunks, error_str

    def _sop_format_report(self, knowledge_chunks: List[KnowledgeChunk]) -> str:
        logger.debug(f"Agent {self.agent_config.agent_id}: SOP Step - Formatting Report from {len(knowledge_chunks)} chunks...")
        formatted_findings = f"Research Report - {len(knowledge_chunks)} finding(s):\n\n"
        if not knowledge_chunks:
            formatted_findings += "No specific knowledge chunks were found during the research.\n"
            return formatted_findings

        for i, chunk in enumerate(knowledge_chunks):
            formatted_findings += f"Finding {i+1}: {chunk.content}\n"
            if chunk.source_result_ids:
                 formatted_findings += f"Sources: {', '.join(chunk.source_result_ids)}\n\n"
            else:
                formatted_findings += "Sources: Not specified\n\n"
        logger.debug(f"Agent {self.agent_config.agent_id}: SOP Step - Report Formatting Complete.")
        return formatted_findings.strip()

    async def _sop_save_report_to_file(self, report_content: str, task_description: str, context: TaskContext) -> Optional[str]:
        logger.debug(f"Agent {self.agent_config.agent_id}: SOP Step - Saving Report to File...")
        error_str: Optional[str] = None
        task_desc_lower = task_description.lower()

        if "save report" in task_desc_lower or "write report to file" in task_desc_lower:
            path_keyword = "save report to" if "save report to" in task_desc_lower else "write report to file"
            parsed_path = _parse_path_from_description(task_description, path_keyword)

            if parsed_path:
                logger.info(f"Attempting to save research report to: {parsed_path}")
                write_tool_output: ToolOutput = await self._use_tool("file_write", {"file_path": parsed_path, "content": report_content})
                if write_tool_output.error:
                    error_str = f"Failed to write report to '{parsed_path}': {write_tool_output.error} (Code: {write_tool_output.status_code})"
                    logger.warn(error_str)
                else:
                    logger.info(f"Research report successfully saved to {parsed_path}.")
            else:
                logger.info("'save report' mentioned but path not parsable. Report not saved to file.")
                # Not an error for the SOP step itself if path is just unparsable, but could be logged.
        else:
            logger.info("No instruction to save report to file found in task description.")

        logger.debug(f"Agent {self.agent_config.agent_id}: SOP Step - Save Report to File Complete.")
        return error_str


    async def execute_task(self, task: Task, context: TaskContext) -> TaskResult:
        current_task_obj = context.current_task
        task_id_to_report = current_task_obj.task_id
        original_task_description = current_task_obj.description

        logger.info(f"ResearcherAgent ({self.agent_config.agent_id}) starting SOP for task: {task_id_to_report} - {original_task_description}")

        if not original_task_description:
            logger.error(f"Task {task_id_to_report} has no description.")
            return TaskResult(task_id=task_id_to_report, status=TaskStatus.FAILED, error_message="Task has no description for research.")

        # SOP Step 1: Clarify Research Goal
        clarified_goal = await self._sop_clarify_research_goal(original_task_description, context)

        # SOP Step 2: Execute Research Cycle
        knowledge_chunks, research_err = await self._sop_execute_research_cycle(clarified_goal, context)

        if research_err:
            logger.error(f"Research cycle failed for task {task_id_to_report}: {research_err}")
            return TaskResult(task_id=task_id_to_report, status=TaskStatus.FAILED, error_message=research_err, output=[])

        if not knowledge_chunks: # Can be None if error, or empty list if no results
            logger.info(f"Research yielded no knowledge chunks for task {task_id_to_report}.")
            return TaskResult(task_id=task_id_to_report, status=TaskStatus.COMPLETED, output=[], error_message="Research yielded no specific information chunks.")

        # SOP Step 3: Format Report
        formatted_report_content = self._sop_format_report(knowledge_chunks)

        # SOP Step 4: Save Report to File (if requested)
        save_err = await self._sop_save_report_to_file(formatted_report_content, original_task_description, context)

        final_output_payload: Dict[str, Any] = {"knowledge_chunks": knowledge_chunks, "formatted_report": formatted_report_content}
        if save_err:
            # If saving was requested and failed, this is a task failure.
            # The specific error is logged by _sop_save_report_to_file.
            logger.warn(f"Failed to save report for task {task_id_to_report}: {save_err}")
            # Include formatted report in output even if saving failed, but mark task as FAILED.
            return TaskResult(task_id=task_id_to_report, status=TaskStatus.FAILED, output=final_output_payload, error_message=save_err)

        # Check if a path was attempted for saving to include in success message
        path_keyword_save = "save report to" if "save report to" in original_task_description.lower() else None
        path_keyword_write = "write report to file" if "write report to file" in original_task_description.lower() else None
        path_keyword = path_keyword_save or path_keyword_write

        if path_keyword:
            parsed_path_for_message = _parse_path_from_description(original_task_description, path_keyword)
            if parsed_path_for_message:
                 final_output_payload["report_file"] = parsed_path_for_message
                 final_output_payload["message"] = f"Research complete and report saved to '{parsed_path_for_message}'."
            else: # "save report" mentioned but path was not parsable
                 final_output_payload["message"] = "Research complete. Report was to be saved, but file path was unclear."


        logger.info(f"ResearcherAgent SOP for task {task_id_to_report} finished successfully.")
        return TaskResult(
            task_id=task_id_to_report,
            status=TaskStatus.COMPLETED,
            output=final_output_payload
        )

# Type hinting guards
if "AgentConfig" not in globals():
    class AgentConfig: agent_id: str; role_name: str; llm_config: Optional[Dict]
if "TaskContext" not in globals():
    class TaskContext: current_task: Any; project_context: Any
if "TaskStatus" not in globals():
    class TaskStatus: COMPLETED="COMPLETED"; FAILED="FAILED"
if "Task" not in globals():
    class Task: description: str; task_id: str
if "AbstractBaseAgent" not in globals():
    class AbstractBaseAgent: agent_config: AgentConfig; llm_aggregator: Any; tools_registry: Optional[ToolsRegistry]; def __init__(self,ac, llma, tr): pass; async def _use_llm(self, m) -> Any: pass; async def _use_tool(self, tn, ti) -> ToolOutput: pass # type: ignore
if "LLMAggregator" not in globals():
    class LLMAggregator: pass
if "ToolsRegistry" not in globals():
    class ToolsRegistry: pass
if "ResearchQuery" not in globals():
    class ResearchQuery: query_id: str
if "KnowledgeChunk" not in globals():
    class KnowledgeChunk: content: str; source_result_ids: List[str]
if "ProjectContext" not in globals():
    class ProjectContext: project_description: Optional[str]
if "ToolOutput" not in globals():
    class ToolOutput: error: Optional[str]; output: Any; status_code: int
if "ChatMessage" not in globals():
    class ChatMessage: pass
