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
from core.research_structures import ResearchQuery, KnowledgeChunk, WebSearchResult, TavilySearchSessionReport # Added WebSearchResult, TavilySearchSessionReport

# Import new/updated components for code generation
from core.code_generator import MultiLanguageCodeGenerator
from core.generation_structures import CodeSpecification, CodeGenerationResult # CodeGenerationResult for type hint
from core.planning_structures import Task, ExecutionPlan # Added ExecutionPlan for type hint
import httpx # Import httpx for the shared client
import json # For pretty printing dicts

# Settings to get TAVILY_API_KEY
from config import settings


logger = structlog.get_logger(__name__)

async def simulate_workflow(user_instruction: str):
    '''
    Simulates the core planning and reasoning workflow.
    '''
    logger.info("Starting simulation: C Language Support Demo", instruction=user_instruction)
    # ... (start_time_total, Tavily API Key handling, LLMAggregator setup as before) ...
    # ... (Planner, Reasoner, StateTracker instantiation as before) ...
    # ... (Research components instantiation as before) ...
    # ... (MultiLanguageCodeGenerator instantiation as before - it now supports C) ...
    # No change needed for component instantiation, as MultiLanguageCodeGenerator itself was updated.
    start_time_total = time.monotonic()

    # Load Tavily API Key from settings
    tavily_api_key = settings.TAVILY_API_KEY
    if not tavily_api_key:
        logger.warn("TAVILY_API_KEY not found in environment/settings. Tavily search will fail if triggered.")

    try:
        account_manager = AccountManager()
        mock_provider_configs = {"mock_provider_1": OpenHandsProviderConfig(name="mock_provider_1", status=ProviderStatus.ACTIVE, models=[], credentials=[])}
        router = ProviderRouter(provider_configs=mock_provider_configs)
        rate_limiter = RateLimiter()
        llm_aggregator = LLMAggregator(providers=[], account_manager=account_manager, router=router, rate_limiter=rate_limiter)

        planner = DevikaInspiredPlanner(llm_aggregator=llm_aggregator)
        reasoner = ContextualReasoningEngine(llm_aggregator=llm_aggregator)
        state_tracker = StateTracker()

        keyword_extractor = ContextualKeywordExtractor(llm_aggregator=llm_aggregator)
        web_researcher = WebResearcher(tavily_api_key=tavily_api_key or "NO_KEY_PROVIDED_TAVILY")
        relevance_scorer = RelevanceScorer(llm_aggregator=llm_aggregator)
        research_assistant = IntelligentResearchAssistant(
            keyword_extractor=keyword_extractor, web_researcher=web_researcher,
            relevance_scorer=relevance_scorer, llm_aggregator=llm_aggregator )

        code_generator = MultiLanguageCodeGenerator(llm_aggregator=llm_aggregator)
        logger.info("All components initialized for C language support simulation.")

    except Exception as e: # Catch errors during component initialization
        logger.error("Failed to initialize core components for simulation", error=str(e), exc_info=True)
        return


    project_ctx = ProjectContext(project_name="CSimulationProject",
                                 project_description="Simulating C code generation with cppcheck integration.")
    current_plan: Optional[ExecutionPlan] = None

    try:
        # ... (Intent Parsing and Task Decomposition as before) ...
        logger.info("Step 1: Parsing user intent...")
        parsed_intent = await planner.parse_user_intent(user_instruction, context=project_ctx)

        logger.info("Step 2: Decomposing task into execution plan...")
        current_plan = await planner.decompose_complex_task(parsed_intent, context=project_ctx)
        state_tracker.start_plan(current_plan.plan_id, user_instruction, len(current_plan.tasks))


        if not current_plan.tasks:
            # ... (handling of no tasks as before) ...
            logger.warn("No tasks generated in the execution plan.", plan_id=current_plan.plan_id)
            current_plan.overall_status = TaskStatus.COMPLETED # type: ignore
        else:
            logger.info(f"Step 3: Processing {len(current_plan.tasks)} tasks in the plan...", plan_id=current_plan.plan_id)
            all_tasks_ultimately_successful = True

            for task_index, task in enumerate(current_plan.tasks):
                # ... (task processing setup: logging, state_tracker.start_task, status update) ...
                logger.info(f"Processing task {task_index + 1}/{len(current_plan.tasks)}", task_id=task.task_id, description=task.description)
                state_tracker.start_task(current_plan.plan_id, task.task_id, task.description, task.dependencies)
                task.status = TaskStatus.IN_PROGRESS # type: ignore

                # ... (Context Analysis, Reasoning, Decision Making as before) ...
                analyzed_context = await reasoner.analyze_context(task, project_context=project_ctx)
                reasoning_output = await reasoner.reason_about_task(task, analyzed_context)
                decision = reasoner.make_decision(reasoning_output)
                state_tracker.update_task_reasoning(current_plan.plan_id, task.task_id, decision)
                task.reasoning_log.append(f"InitialReasoning: {decision}")


                # --- Updated Research Trigger Logic & Logging ---
                research_data: Optional[Dict[str, Any]] = None
                needs_research_keywords = ["research", "find information", "investigate", "what is", "how to", "learn about", "explore", "tavily search for"]
                trigger_research = any(kw in task.description.lower() for kw in needs_research_keywords) or \
                                   (decision.get('action') == "NEEDS_CLARIFICATION" and decision.get('confidence', 1.0) < 0.7)
                if trigger_research:
                    if not tavily_api_key:
                        logger.error("Skipping research: TAVILY_API_KEY is not set.", task_id=task.task_id)
                        task.reasoning_log.append("ResearchSkipped: TAVILY_API_KEY missing.")
                    else:
                        logger.info("Task flagged for research with Tavily, invoking IntelligentResearchAssistant.", task_id=task.task_id)
                        project_summary_for_research = project_ctx.project_description if project_ctx else None
                        current_research_query = ResearchQuery(
                            original_task_description=task.description, # Or task.raw_instruction or combined
                            project_context_summary=project_summary_for_research
                        )
                        try:
                            research_data = await research_assistant.research_for_task(current_research_query)

                            if research_data:
                                tav_report: Optional[TavilySearchSessionReport] = research_data.get("session_report")
                                k_chunks: List[KnowledgeChunk] = research_data.get("knowledge_chunks", [])
                                all_web_res: List[WebSearchResult] = research_data.get("all_web_search_results", [])
                                processed_for_synth: List[ProcessedResult] = research_data.get("processed_results_for_synthesis", []) # type: ignore

                                if tav_report:
                                    logger.info("Tavily research session report",
                                                query_echo=tav_report.query_echo,
                                                answer=tav_report.overall_answer,
                                                num_raw_tavily_results=tav_report.num_results_returned,
                                                session_metadata=tav_report.session_metadata)

                                if k_chunks:
                                    logger.info("Research completed, knowledge chunks obtained.",
                                                task_id=task.task_id, num_chunks=len(k_chunks))
                                    task.reasoning_log.append(f"KnowledgeChunks: {[chunk.content for chunk in k_chunks]}")
                                    if processed_for_synth:
                                        log_synth_sources = [{"url": pr.source_web_search_result.url, "local_score": pr.local_relevance_score, "original_score": pr.source_web_search_result.score} for pr in processed_for_synth]
                                        logger.debug("Sources for synthesis", sources_count=len(log_synth_sources) ,sources_preview=json.dumps(log_synth_sources[:2], indent=2)) # Log first 2
                                else:
                                    logger.info("Research completed, but no knowledge chunks were synthesized.", task_id=task.task_id)
                        except Exception as e_research:
                            logger.error("Error during Tavily research execution for task", task_id=task.task_id, error=str(e_research), exc_info=True)
                            task.reasoning_log.append(f"ResearchFailedError: {str(e_research)}")
                # --- End Updated Research Logic ---

                # --- Code Generation Trigger Logic (now potentially for C) ---
                generated_code_info: Optional[CodeGenerationResult] = None
                # Keywords to trigger C code generation
                needs_c_code_keywords = ["generate c code", "write c function", "implement in c", "c program for"]
                # Python keywords kept for flexibility if instruction is mixed
                needs_python_code_keywords = ["generate python script", "python function for", "python class for"]

                target_lang_for_gen = None
                if any(kw in task.description.lower() for kw in needs_c_code_keywords):
                    target_lang_for_gen = "c"
                elif any(kw in task.description.lower() for kw in needs_python_code_keywords): # Fallback or specific request
                    target_lang_for_gen = "python"

                # More generic trigger if task description implies coding without specifying language explicitly
                # For now, rely on explicit keywords or a more advanced planner decision in future.

                if target_lang_for_gen:
                    logger.info(f"Task flagged for {target_lang_for_gen.upper()} code generation.", task_id=task.task_id)
                    code_spec = CodeSpecification(
                        target_language=target_lang_for_gen,
                        prompt_details=task.description,
                        context_summary=project_ctx.project_description,
                        constraints=[f"Ensure the {target_lang_for_gen} code is robust and follows best practices provided."]
                    )
                    try:
                        generated_code_info = await code_generator.generate_code(code_spec)
                        if generated_code_info and generated_code_info.succeeded:
                            logger.info(f"{target_lang_for_gen.upper()} code generation successful.",
                                        task_id=task.task_id,
                                        code_snippet=(generated_code_info.generated_code or "")[:150] + "...",
                                        quality_score=generated_code_info.quality_score)
                            if generated_code_info.issues_found:
                                logger.warn(f"Quality issues found in generated {target_lang_for_gen.upper()} code:",
                                            task_id=task.task_id,
                                            num_issues=len(generated_code_info.issues_found),
                                            issues=json.dumps(generated_code_info.issues_found, indent=2))

                            task.reasoning_log.append(
                                f"Generated{target_lang_for_gen.upper()}CodeInfo: Score={generated_code_info.quality_score}, "
                                f"NumIssues={len(generated_code_info.issues_found)}. Code: {(generated_code_info.generated_code or '')[:100]}..."
                            )
                            task.output = generated_code_info.generated_code
                        elif generated_code_info:
                            logger.warn(f"{target_lang_for_gen.upper()} code generation attempt failed or had issues.",
                                        task_id=task.task_id,
                                        error=generated_code_info.error_message,
                                        num_issues=len(generated_code_info.issues_found),
                                        issues_preview=json.dumps(generated_code_info.issues_found[:2], indent=2),
                                        score=generated_code_info.quality_score)
                            task.reasoning_log.append(f"CodeGenerationFailed ({target_lang_for_gen.upper()}): {generated_code_info.error_message}, Issues: {len(generated_code_info.issues_found)}")
                            all_tasks_ultimately_successful = False
                    except Exception as e_codegen:
                        logger.error(f"Critical error during {target_lang_for_gen.upper()} code generation call for task", task_id=task.task_id, error=str(e_codegen), exc_info=True)
                        task.reasoning_log.append(f"CodeGenSystemError ({target_lang_for_gen.upper()}): {str(e_codegen)}")
                        all_tasks_ultimately_successful = False
                # --- End Code Generation ---

                # ... (Task status update logic as before, considering codegen success) ...
                if decision.get('action') == "PROCEED":
                    if target_lang_for_gen and (not generated_code_info or not generated_code_info.succeeded):
                        task_status = TaskStatus.FAILED
                        task_message = f"{target_lang_for_gen.upper()} code generation was required but failed or had significant quality issues."
                        all_tasks_ultimately_successful = False
                    else:
                        task_status = TaskStatus.COMPLETED
                        task_message = "Task simulated successfully."
                        # ... (add research/codegen notes to message as before) ...

                    task_result = TaskResult(task_id=task.task_id, status=task_status, message=task_message) # type: ignore
                    task.status = task_status # type: ignore
                    state_tracker.complete_task(current_plan.plan_id, task.task_id, task.status.value, task_result.message, output_summary=str(task.output)[:100] if task.output else None)
                else: # NEEDS_CLARIFICATION
                    task.status = TaskStatus.CLARIFICATION_NEEDED # type: ignore
                    all_tasks_ultimately_successful = False
                    state_tracker.complete_task(current_plan.plan_id, task.task_id, task.status.value, decision.get('details'))

            # ... (Overall plan status update as before) ...
            if current_plan: # Ensure current_plan exists
                if all_tasks_ultimately_successful and all(t.status == TaskStatus.COMPLETED for t in current_plan.tasks if t.status != TaskStatus.CLARIFICATION_NEEDED):
                    current_plan.overall_status = TaskStatus.COMPLETED # type: ignore
                elif any(t.status == TaskStatus.FAILED for t in current_plan.tasks):
                    current_plan.overall_status = TaskStatus.FAILED # type: ignore
                else:
                    current_plan.overall_status = TaskStatus.CLARIFICATION_NEEDED # type: ignore

        total_execution_time = time.monotonic() - start_time_total
        if current_plan:
            state_tracker.complete_plan(current_plan.plan_id, current_plan.overall_status.value, total_execution_time) # type: ignore
            logger.info("Workflow simulation finished (C Lang Support).", plan_id=current_plan.plan_id, overall_status=current_plan.overall_status.value, duration_seconds=round(total_execution_time, 2)) # type: ignore
        else:
            logger.error("Simulation ended prematurely, no plan was fully created.")


    except Exception as e:
        # ... (Critical error handling as before) ...
        logger.error("Critical error during workflow simulation", error=str(e), exc_info=True)
        if current_plan and current_plan.plan_id:
             state_tracker.complete_plan(current_plan.plan_id, TaskStatus.FAILED.value, time.monotonic() - start_time_total)
    finally:
        # ... (Cleanup as before) ...
        if 'web_researcher' in locals() and hasattr(web_researcher, 'close_client'):
            await web_researcher.close_client()
        logger.info("Simulation cleanup attempted.")


if __name__ == "__main__":
    # ... (TAVILY_API_KEY check as before) ...
    if not settings.TAVILY_API_KEY:
        logger.warn("TAVILY_API_KEY environment variable not set. Research functionality will be impacted.")

    logger.info("NOTE: For C language quality checks to work fully, 'cppcheck' must be installed and accessible in the system PATH.")

    sample_instruction_c_code = "Generate a C function that calculates the factorial of an integer. It should handle negative inputs by returning -1. Ensure proper header includes like stdio.h for potential printf usage if you add main for testing."
    # sample_instruction_flake8_test = "..." # (previous python test instruction)

    current_instruction_to_run = sample_instruction_c_code

    logger.info(f"Running simulation with instruction: '{current_instruction_to_run}'")
    asyncio.run(simulate_workflow(current_instruction_to_run))
