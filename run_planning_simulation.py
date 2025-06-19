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
    logger.info("Starting planning workflow simulation with Tavily research and enhanced code gen", instruction=user_instruction)
    start_time_total = time.monotonic()

    logger.info("Starting planning workflow simulation with Tavily research and enhanced code gen", instruction=user_instruction)
    start_time_total = time.monotonic()

    # Load Tavily API Key from settings
    tavily_api_key = settings.TAVILY_API_KEY
    if not tavily_api_key:
        logger.warn("TAVILY_API_KEY not found in environment/settings. Tavily search will likely fail or be skipped.")

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
        return
    # --- End Simplified Setup ---

    # Instantiate core components
    planner = DevikaInspiredPlanner(llm_aggregator=llm_aggregator)
    reasoner = ContextualReasoningEngine(llm_aggregator=llm_aggregator)
    state_tracker = StateTracker()

    # Instantiate research components
    keyword_extractor = ContextualKeywordExtractor(llm_aggregator=llm_aggregator)
    web_researcher = WebResearcher(tavily_api_key=tavily_api_key or "NO_KEY_PROVIDED_SIMULATION")
    relevance_scorer = RelevanceScorer(llm_aggregator=llm_aggregator)
    research_assistant = IntelligentResearchAssistant(
        keyword_extractor=keyword_extractor,
        web_researcher=web_researcher,
        relevance_scorer=relevance_scorer,
        llm_aggregator=llm_aggregator
    )
    logger.info("IntelligentResearchAssistant initialized for simulation (with Tavily-based WebResearcher).")

    # MultiLanguageCodeGenerator now internally instantiates BestPracticesDatabase,
    # PythonAnalyzer, and CodeQualityChecker (with Flake8).
    code_generator = MultiLanguageCodeGenerator(llm_aggregator=llm_aggregator)
    logger.info("MultiLanguageCodeGenerator initialized (with Flake8-enhanced QualityChecker).")

    project_ctx = ProjectContext(project_name="ProdGradeCodeGenSimProject",
                                 project_description="Simulating Python code generation with Flake8 checks.")
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

                # --- Updated Code Generation Trigger Logic & Enhanced Logging ---
                generated_code_info: Optional[CodeGenerationResult] = None
                needs_code_keywords = ["generate code", "write a script", "implement function", "create class", "python code for", "develop python"]
                trigger_code_generation = any(kw in task.description.lower() for kw in needs_code_keywords)

                if trigger_code_generation:
                    logger.info("Task flagged for Python code generation (with Flake8 quality checks).", task_id=task.task_id)
                    code_spec = CodeSpecification(
                        target_language="python",
                        prompt_details=task.description,
                        context_summary=project_ctx.project_description,
                        constraints=["Ensure the code is robust. Follow Python best practices provided."]
                    )
                    try:
                        generated_code_info = await code_generator.generate_code(code_spec)
                        if generated_code_info and generated_code_info.succeeded:
                            # Count Flake8 and custom issues
                            flake8_issue_count = sum(1 for issue in generated_code_info.issues_found if issue.get("type") == "flake8")
                            custom_issue_count = sum(1 for issue in generated_code_info.issues_found if issue.get("type") == "custom")

                            logger.info("Python code generation successful.",
                                        task_id=task.task_id,
                                        code_snippet=(generated_code_info.generated_code or "")[:150] + "...",
                                        quality_score=generated_code_info.quality_score,
                                        flake8_issues=flake8_issue_count,
                                        custom_issues=custom_issue_count,
                                        total_issues=len(generated_code_info.issues_found)
                                        )
                            if generated_code_info.issues_found:
                                # Log all issues found, which are now dicts
                                logger.warn("Quality issues found in generated code (Flake8 + custom):",
                                            task_id=task.task_id,
                                            num_issues=len(generated_code_info.issues_found),
                                            issues=json.dumps(generated_code_info.issues_found, indent=2))

                            task.reasoning_log.append(
                                f"GeneratedCodeInfo: Score={generated_code_info.quality_score}, "
                                f"NumIssues={len(generated_code_info.issues_found)}. Code: {(generated_code_info.generated_code or '')[:100]}..."
                            )
                            task.output = generated_code_info.generated_code
                        elif generated_code_info:
                            logger.warn("Python code generation attempt failed or had issues.",
                                        task_id=task.task_id,
                                        error=generated_code_info.error_message,
                                        num_issues=len(generated_code_info.issues_found),
                                        # Log issues directly as they are now dicts
                                        issues_found_details=json.dumps(generated_code_info.issues_found, indent=2),
                                        score=generated_code_info.quality_score)
                            task.reasoning_log.append(f"CodeGenerationFailed: {generated_code_info.error_message}, Issues: {len(generated_code_info.issues_found)}")
                            all_tasks_ultimately_successful = False
                    except Exception as e_codegen:
                        logger.error("Critical error during code generation call for task", task_id=task.task_id, error=str(e_codegen), exc_info=True)
                        task.reasoning_log.append(f"CodeGenSystemError: {str(e_codegen)}")
                        all_tasks_ultimately_successful = False
                # --- End Updated Code Generation ---

                # ... (Task status update logic, adjusted for codegen success/failure) ...
                if decision.get('action') == "PROCEED":
                    if trigger_code_generation and (not generated_code_info or not generated_code_info.succeeded):
                        task_status = TaskStatus.FAILED
                        task_message = "Code generation was required but failed or had significant quality issues."
                        all_tasks_ultimately_successful = False
                    else:
                        task_status = TaskStatus.COMPLETED
                        task_message = "Task simulated successfully."
                        if research_data and research_data.get("knowledge_chunks"): task_message += f" (Research conducted)"
                        if generated_code_info and generated_code_info.succeeded: task_message += " (Code generated)."

                    task_result = TaskResult(task_id=task.task_id, status=task_status, message=task_message) # type: ignore
                    task.status = task_status # type: ignore
                    state_tracker.complete_task(current_plan.plan_id, task.task_id, task.status.value, task_result.message, output_summary=str(task.output)[:100] if task.output else None)
                else: # NEEDS_CLARIFICATION
                    task.status = TaskStatus.CLARIFICATION_NEEDED # type: ignore
                    all_tasks_ultimately_successful = False
                    state_tracker.complete_task(current_plan.plan_id, task.task_id, task.status.value, decision.get('details'))

            # ... (Overall plan status update as before) ...
            if current_plan: # Ensure current_plan is not None
                if all_tasks_ultimately_successful and all(t.status == TaskStatus.COMPLETED for t in current_plan.tasks if t.status != TaskStatus.CLARIFICATION_NEEDED):
                    current_plan.overall_status = TaskStatus.COMPLETED # type: ignore
                elif any(t.status == TaskStatus.FAILED for t in current_plan.tasks):
                    current_plan.overall_status = TaskStatus.FAILED # type: ignore
                else:
                    current_plan.overall_status = TaskStatus.CLARIFICATION_NEEDED # type: ignore


        total_execution_time = time.monotonic() - start_time_total
        if current_plan:
            state_tracker.complete_plan(current_plan.plan_id, current_plan.overall_status.value, total_execution_time) # type: ignore
            logger.info("Workflow simulation finished (Prod-Grade Python CodeGen).", plan_id=current_plan.plan_id, overall_status=current_plan.overall_status.value, duration_seconds=round(total_execution_time, 2)) # type: ignore
        else:
            logger.error("Simulation ended prematurely, no plan was fully created.")


    except Exception as e:
        logger.error("Critical error during workflow simulation", error=str(e), exc_info=True)
        if current_plan and current_plan.plan_id:
             state_tracker.complete_plan(current_plan.plan_id, TaskStatus.FAILED.value, time.monotonic() - start_time_total)
    finally:
        # Close WebResearcher's http_client if it has one and it was created by WebResearcher
        # The current WebResearcher init takes an API key and creates its own clients.
        # It has a close_client method for its httpx client.
        if 'web_researcher' in locals() and hasattr(web_researcher, 'close_client'):
            await web_researcher.close_client()
        logger.info("Simulation cleanup attempted (e.g., HTTP clients).")


        if 'web_researcher' in locals() and hasattr(web_researcher, 'close_client'):
            await web_researcher.close_client()
        logger.info("Simulation cleanup attempted.")


if __name__ == "__main__":
    if not settings.TAVILY_API_KEY:
        logger.warn("TAVILY_API_KEY environment variable not set. Research functionality will be impacted.")

    sample_instruction_flake8_test = """
    Generate a Python script that does the following:
    1. Defines a function `calculateSum` that takes two arguments `a` and `b` and returns their sum. (Use camelCase for function name for testing)
    2. Defines a class `MyTestData` with a method `get_info` that returns a static string. (Method name also not snake_case for testing)
    3. Has an unused import like `import os, sys`
    4. Contains a line that is excessively long, over 100 characters: print('This is a very very very very very very very very very very very very very very very very very very very very long line of text for testing line length issues.')
    5. Includes a TODO comment.
    6. A function with no docstring: def no_doc_func(): pass
    7. A function that is too long:
    def very_long_function_example():
        x = 1; print(x); x = 2; print(x); x = 3; print(x); x = 4; print(x); x = 5; print(x);
        x = 1; print(x); x = 2; print(x); x = 3; print(x); x = 4; print(x); x = 5; print(x);
        x = 1; print(x); x = 2; print(x); x = 3; print(x); x = 4; print(x); x = 5; print(x);
        x = 1; print(x); x = 2; print(x); x = 3; print(x); x = 4; print(x); x = 5; print(x);
        x = 1; print(x); x = 2; print(x); x = 3; print(x); x = 4; print(x); x = 5; print(x);
        x = 1; print(x); x = 2; print(x); x = 3; print(x); x = 4; print(x); x = 5; print(x); # This will make it > 50 lines for CodeChecker
    """

    current_instruction_to_run = sample_instruction_flake8_test

    logger.info(f"Running simulation with instruction: '{current_instruction_to_run}'")
    asyncio.run(simulate_workflow(current_instruction_to_run))
