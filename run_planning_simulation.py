import asyncio
import os
import sys
import time # For simulating execution time

# Ensure src directory is in Python path
# This might be needed if running the script directly from the root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from config.logging_config import setup_logging # Assuming this path
setup_logging() # Initialize logging early

import structlog

from core.planning_structures import ProjectContext, TaskStatus, TaskResult
from core.planner import DevikaInspiredPlanner
from core.reasoner import ContextualReasoningEngine
from core.state_tracker import StateTracker

# For LLMAggregator instantiation, we need to set up its dependencies.
# This is a simplified setup for the simulation script.
# In a full application, these would be managed more robustly (e.g., via DI or a central app context).
from core.aggregator import LLMAggregator
from core.account_manager import AccountManager
from core.router import ProviderRouter # ProviderRouter needs provider_configs
from core.rate_limiter import RateLimiter
# Mock provider configs for ProviderRouter as we aren't making real calls that need routing here
# but the objects might be expected by constructors.
from models import ProviderConfig as OpenHandsProviderConfig, ProviderStatus

# Import IntelligentResearchAssistant and its components, and ResearchQuery
from core.researcher import IntelligentResearchAssistant
from core.research_components import ContextualKeywordExtractor, WebResearcher, RelevanceScorer
from core.research_structures import ResearchQuery, KnowledgeChunk # KnowledgeChunk for type hint

# Import new/updated components for code generation
from core.code_generator import MultiLanguageCodeGenerator
from core.generation_structures import CodeSpecification, CodeGenerationResult # CodeGenerationResult for type hint
from core.planning_structures import Task # Task needed for type hint
import httpx # Import httpx for the shared client


logger = structlog.get_logger(__name__)

async def simulate_workflow(user_instruction: str):
    '''
    Simulates the core planning and reasoning workflow.
    '''
    logger.info("Starting planning workflow simulation with live research and enhanced code gen", instruction=user_instruction)
    start_time_total = time.monotonic()

    # --- Shared HTTP Client for components that need it ---
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as shared_http_client:
        logger.info("Shared httpx.AsyncClient initialized.")

        # --- Simplified Setup for LLMAggregator ---
        try:
            account_manager = AccountManager()
            mock_provider_configs = {"mock_provider_1": OpenHandsProviderConfig(name="mock_provider_1", status=ProviderStatus.ACTIVE, models=[], credentials=[])} # type: ignore
            router = ProviderRouter(provider_configs=mock_provider_configs)
            rate_limiter = RateLimiter()
            llm_aggregator = LLMAggregator(providers=[], account_manager=account_manager, router=router, rate_limiter=rate_limiter)
            logger.info("LLMAggregator initialized for simulation (with no actual providers).")
        except Exception as e:
            logger.error("Failed to initialize LLMAggregator and its dependencies", error=str(e), exc_info=True)
            # Ensure shared_http_client is closed if it was opened before this error.
            # However, it's managed by async with, so it will be handled.
            return
        # --- End Simplified Setup ---

        # Instantiate core components
        planner = DevikaInspiredPlanner(llm_aggregator=llm_aggregator)
        reasoner = ContextualReasoningEngine(llm_aggregator=llm_aggregator)
        state_tracker = StateTracker()

        # Instantiate research components with shared client
        keyword_extractor = ContextualKeywordExtractor(llm_aggregator=llm_aggregator)
        web_researcher = WebResearcher(http_client=shared_http_client)
        relevance_scorer = RelevanceScorer(llm_aggregator=llm_aggregator)
        research_assistant = IntelligentResearchAssistant(
            keyword_extractor=keyword_extractor,
            web_researcher=web_researcher,
            relevance_scorer=relevance_scorer,
            llm_aggregator=llm_aggregator
        )
        logger.info("IntelligentResearchAssistant initialized for simulation (with live WebResearcher).")

        # Instantiate code generation components
        code_generator = MultiLanguageCodeGenerator(llm_aggregator=llm_aggregator)
        logger.info("MultiLanguageCodeGenerator initialized for simulation.")


        project_ctx = ProjectContext(project_name="LiveSimProject", project_description="Project simulating live research and enhanced code gen.")
        current_plan: Optional[Any] = None # ExecutionPlan is defined in planning_structures, but using Any for flexibility here

        try:
            logger.info("Step 1: Parsing user intent...")
            parsed_intent = await planner.parse_user_intent(user_instruction, context=project_ctx)

            logger.info("Step 2: Decomposing task into execution plan...")
            current_plan = await planner.decompose_complex_task(parsed_intent, context=project_ctx)
            state_tracker.start_plan(current_plan.plan_id, user_instruction, len(current_plan.tasks))

            if not current_plan.tasks:
                logger.warn("No tasks generated in the execution plan.", plan_id=current_plan.plan_id)
                current_plan.overall_status = TaskStatus.COMPLETED
            else:
                logger.info(f"Step 3: Processing {len(current_plan.tasks)} tasks in the plan...", plan_id=current_plan.plan_id)
                all_tasks_ultimately_successful = True

                for task_index, task in enumerate(current_plan.tasks):
                    logger.info(f"Processing task {task_index + 1}/{len(current_plan.tasks)}", task_id=task.task_id, description=task.description)
                    state_tracker.start_task(current_plan.plan_id, task.task_id, task.description, task.dependencies)
                    task.status = TaskStatus.IN_PROGRESS

                    analyzed_context = await reasoner.analyze_context(task, project_context=project_ctx)
                    reasoning_output = await reasoner.reason_about_task(task, analyzed_context)
                    decision = reasoner.make_decision(reasoning_output)
                    state_tracker.update_task_reasoning(current_plan.plan_id, task.task_id, decision)
                    task.reasoning_log.append(f"InitialReasoning: {decision}")

                    task_knowledge_chunks: List[KnowledgeChunk] = []
                    needs_research_keywords = ["research", "find information", "investigate", "what is", "how to", "learn about", "explore"]
                    trigger_research = any(kw in task.description.lower() for kw in needs_research_keywords) or \
                                       (decision.get('action') == "NEEDS_CLARIFICATION" and decision.get('confidence', 1.0) < 0.7)

                    if trigger_research:
                        logger.info("Task flagged for live research, invoking IntelligentResearchAssistant.", task_id=task.task_id)
                        project_summary_for_research = project_ctx.project_description if project_ctx else None
                        current_research_query = ResearchQuery(
                            original_task_description=task.description,
                            project_context_summary=project_summary_for_research
                        )
                        try:
                            task_knowledge_chunks = await research_assistant.research_for_task(current_research_query)
                            if task_knowledge_chunks:
                                logger.info("Live research completed, knowledge chunks obtained.", task_id=task.task_id, num_chunks=len(task_knowledge_chunks))
                                task.reasoning_log.append(f"Research Chunks: {[chunk.content for chunk in task_knowledge_chunks]}")
                            else:
                                logger.info("Live research completed, but no knowledge chunks were synthesized.", task_id=task.task_id)
                        except Exception as e_research:
                            logger.error("Error during live research execution for task", task_id=task.task_id, error=str(e_research), exc_info=True)
                            task.reasoning_log.append(f"ResearchFailedError: {str(e_research)}")

                    needs_code_keywords = ["generate code", "write a script", "implement function", "create class", "python code for"]
                    trigger_code_generation = any(kw in task.description.lower() for kw in needs_code_keywords)

                    generated_code_info: Optional[CodeGenerationResult] = None
                    if trigger_code_generation:
                        logger.info("Task flagged for code generation.", task_id=task.task_id)
                        code_spec = CodeSpecification(
                            target_language="python",
                            prompt_details=task.description,
                            context_summary=project_ctx.project_description,
                            constraints=["Ensure the code is well-commented."]
                        )
                        try:
                            generated_code_info = await code_generator.generate_code(code_spec)
                            if generated_code_info and generated_code_info.succeeded:
                                logger.info("Code generation successful for task.", task_id=task.task_id, code_length=len(generated_code_info.generated_code or ""), issues=len(generated_code_info.issues_found), score=generated_code_info.quality_score)
                                task.reasoning_log.append(f"GeneratedCodeInfo: {generated_code_info.generated_code[:100] if generated_code_info.generated_code else 'N/A'}..., Issues: {generated_code_info.issues_found}, Score: {generated_code_info.quality_score}")
                                task.output = generated_code_info.generated_code
                            elif generated_code_info:
                                logger.warn("Code generation attempt failed or had issues for task.", task_id=task.task_id, error=generated_code_info.error_message, issues=generated_code_info.issues_found)
                                task.reasoning_log.append(f"CodeGenerationFailed: {generated_code_info.error_message}, Issues: {generated_code_info.issues_found}")
                                all_tasks_ultimately_successful = False
                        except Exception as e_codegen:
                            logger.error("Error during code generation execution for task", task_id=task.task_id, error=str(e_codegen), exc_info=True)
                            task.reasoning_log.append(f"CodeGenSystemError: {str(e_codegen)}")
                            all_tasks_ultimately_successful = False


                    if decision.get('action') == "PROCEED":
                        if trigger_code_generation and (not generated_code_info or not generated_code_info.succeeded):
                            task_status = TaskStatus.FAILED
                            task_message = "Code generation was required but failed or had issues."
                            all_tasks_ultimately_successful = False
                        else:
                            task_status = TaskStatus.COMPLETED
                            task_message = "Task simulated successfully."
                            if task_knowledge_chunks: task_message += f" (Research found {len(task_knowledge_chunks)} chunks)."
                            if generated_code_info and generated_code_info.succeeded: task_message += " (Code generated)."

                        task_result = TaskResult(task_id=task.task_id, status=task_status, message=task_message)
                        task.status = task_status
                        state_tracker.complete_task(current_plan.plan_id, task.task_id, task.status.value, task_result.message, output_summary=str(task.output)[:100] if task.output else None)
                    else:
                        task_result = TaskResult(task_id=task.task_id, status=TaskStatus.CLARIFICATION_NEEDED, message=decision.get('details'))
                        task.status = TaskStatus.CLARIFICATION_NEEDED
                        all_tasks_ultimately_successful = False
                        state_tracker.complete_task(current_plan.plan_id, task.task_id, task.status.value, task_result.message)

                if all_tasks_ultimately_successful and all(t.status == TaskStatus.COMPLETED for t in current_plan.tasks if t.status != TaskStatus.CLARIFICATION_NEEDED):
                    current_plan.overall_status = TaskStatus.COMPLETED
                elif any(t.status == TaskStatus.FAILED for t in current_plan.tasks):
                    current_plan.overall_status = TaskStatus.FAILED
                else:
                    current_plan.overall_status = TaskStatus.CLARIFICATION_NEEDED

            total_execution_time = time.monotonic() - start_time_total
            if current_plan: # Ensure current_plan is not None
                state_tracker.complete_plan(current_plan.plan_id, current_plan.overall_status.value, total_execution_time)
                logger.info("Workflow simulation finished.", plan_id=current_plan.plan_id, overall_status=current_plan.overall_status.value, duration_seconds=round(total_execution_time,2))
            else: # Should not happen if decomposition was successful
                logger.error("Execution plan is None at the end of simulation. This should not happen if decomposition succeeded.")
                state_tracker._log_event("plan_failed", "unknown_plan_id_at_completion", None, {"reason": "Execution plan object was None."})


        except Exception as e:
            logger.error("Critical error during workflow simulation", error=str(e), exc_info=True)
            if current_plan and current_plan.plan_id:
                 state_tracker.complete_plan(current_plan.plan_id, TaskStatus.FAILED.value, time.monotonic() - start_time_total)
        finally:
            await web_researcher.close_client()
            # await llm_aggregator.close() # Consider if llm_aggregator needs explicit closing
            logger.info("Shared httpx.AsyncClient and other resources closed/handled.")


    # 6. Print session history from state tracker (optional)
    logger.info("\n--- Simulation Event History ---")
    for event in state_tracker.get_session_history():
        # Convert entire event dict to string for a single log call, or let structlog handle dict.
        logger.info("Event", **event) # Pass event dict as kwargs for structlog to expand
    logger.info("--- End of Simulation Event History ---")


if __name__ == "__main__":
    sample_instruction = "Develop a Python script to analyze 'sales_data.csv', calculate total monthly sales, and generate a bar chart visualization. The script should handle potential errors in the CSV file."

    # More instructions for testing:
    # sample_instruction = "Refactor the user authentication module to use OAuth2."
    # sample_instruction = "Set up a new CI/CD pipeline for the web application project."
    # sample_instruction = "What is the capital of France?" # Test a non-task based instruction

    logger.info(f"Running simulation with instruction: '{sample_instruction}'")
    asyncio.run(simulate_workflow(sample_instruction))
