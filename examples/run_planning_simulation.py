import asyncio
import os
import sys
import time # For simulating execution time

# Ensure parent directory is in Python path to access src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.logging_config import setup_logging # Assuming this path
setup_logging() # Initialize logging early

import structlog

from src.core.planning_structures import ProjectContext, TaskStatus, TaskResult
from src.core.planner import DevikaInspiredPlanner
from src.core.reasoner import ContextualReasoningEngine
from src.core.state_tracker import StateTracker

# For LLMAggregator instantiation, we need to set up its dependencies.
# This is a simplified setup for the simulation script.
# In a full application, these would be managed more robustly (e.g., via DI or a central app context).
from src.core.aggregator import LLMAggregator
from src.core.account_manager import AccountManager
from src.core.router import ProviderRouter # ProviderRouter needs provider_configs
from src.core.rate_limiter import RateLimiter
# Mock provider configs for ProviderRouter as we aren't making real calls that need routing here
# but the objects might be expected by constructors.
from src.models import ProviderConfig as OpenHandsProviderConfig, ProviderStatus

# Import IntelligentResearchAssistant and its components, and ResearchQuery
from core.researcher import IntelligentResearchAssistant
from core.research_components import ContextualKeywordExtractor, WebResearcher, RelevanceScorer
from core.research_structures import ResearchQuery, KnowledgeChunk


logger = structlog.get_logger(__name__)

async def simulate_workflow(user_instruction: str):
    '''
    Simulates the core planning and reasoning workflow.
    '''
    logger.info("Starting planning workflow simulation", instruction=user_instruction)
    start_time_total = time.monotonic()

    # --- Simplified Setup for LLMAggregator ---
    # In a real app, LLMAggregator and its dependencies would be part of the application context.
    # For this simulation, we initialize them with minimal viable setup.
    try:
        # 1. AccountManager (relies on OPENHANDS_ENCRYPTION_KEY from settings)
        # Ensure settings are loaded (done by setup_logging indirectly, but good to be aware)
        account_manager = AccountManager() # Uses settings.OPENHANDS_ENCRYPTION_KEY

        # 2. ProviderRouter (needs provider_configs)
        # Create minimal mock provider_configs. In a real app, these come from provider setup.
        mock_provider_configs = {
            "mock_provider_1": OpenHandsProviderConfig(
                name="mock_provider_1",
                status=ProviderStatus.ACTIVE,
                models=[], # type: ignore # No models needed for this simulation
                credentials=[] # type: ignore # No credentials needed for this simulation
            )
        }
        router = ProviderRouter(provider_configs=mock_provider_configs)

        # 3. RateLimiter
        rate_limiter = RateLimiter()

        # 4. LLMAggregator
        # For the simulation, we pass an empty list of actual provider instances.
        # The planner/reasoner use it conceptually for LLM calls.
        # If an actual LLM call is made and no providers are configured/available, it would fail.
        # This simulation primarily tests the flow, not actual LLM interactions unless set up.
        llm_aggregator = LLMAggregator(
            providers=[], # No actual providers needed for this simulation if LLM calls are mocked/abstracted
            account_manager=account_manager,
            router=router,
            rate_limiter=rate_limiter
        )
        logger.info("LLMAggregator initialized for simulation (with no actual providers).")

    except Exception as e:
        logger.error("Failed to initialize LLMAggregator and its dependencies for simulation", error=str(e), exc_info=True)
        logger.error("Please ensure OPENHANDS_ENCRYPTION_KEY is set in your environment if AccountManager fails.")
        logger.error("This simulation may not fully work without a running LLM if not mocked.")
        return
    # --- End Simplified Setup ---


    # Instantiate core components
    planner = DevikaInspiredPlanner(llm_aggregator=llm_aggregator)
    reasoner = ContextualReasoningEngine(llm_aggregator=llm_aggregator) # llm_aggregator for future use by reasoner
    state_tracker = StateTracker()

    # Instantiate research components
    keyword_extractor = ContextualKeywordExtractor(llm_aggregator=llm_aggregator)
    web_researcher = WebResearcher() # Uses mock content, no LLM needed for itself
    relevance_scorer = RelevanceScorer(llm_aggregator=llm_aggregator)

    research_assistant = IntelligentResearchAssistant(
        keyword_extractor=keyword_extractor,
        web_researcher=web_researcher,
        relevance_scorer=relevance_scorer,
        llm_aggregator=llm_aggregator # For synthesis step
    )
    logger.info("IntelligentResearchAssistant initialized for simulation.")


    # 1. Create a ProjectContext (can be minimal for this simulation)
    project_ctx = ProjectContext(project_name="SimulatedProject", project_description="A project for workflow simulation.")
    logger.info("ProjectContext created", project_name=project_ctx.project_name, project_id=project_ctx.project_id)

    # Start tracking the overall plan
    # Plan ID will be generated by ExecutionPlan constructor. For now, we'll get it after plan creation.

    current_plan = None # To store the execution plan

    try:
        # 2. Parse user intent
        logger.info("Step 1: Parsing user intent...")
        parsed_intent = await planner.parse_user_intent(user_instruction, context=project_ctx)
        logger.info("Intent parsing complete", parsed_intent=parsed_intent)

        # 3. Decompose task into an execution plan
        logger.info("Step 2: Decomposing task into execution plan...")
        current_plan = await planner.decompose_complex_task(parsed_intent, context=project_ctx)
        logger.info("Task decomposition complete", plan_id=current_plan.plan_id, num_tasks=len(current_plan.tasks))
        state_tracker.start_plan(current_plan.plan_id, user_instruction, len(current_plan.tasks))


        # 4. Iterate through tasks in the ExecutionPlan
        if not current_plan.tasks:
            logger.warn("No tasks generated in the execution plan.", plan_id=current_plan.plan_id)
            current_plan.overall_status = TaskStatus.COMPLETED # Or FAILED if no tasks from a valid goal is an error
        else:
            logger.info(f"Step 3: Processing {len(current_plan.tasks)} tasks in the plan...", plan_id=current_plan.plan_id)
            all_tasks_successful = True
            for task_index, task in enumerate(current_plan.tasks):
                logger.info(f"Processing task {task_index + 1}/{len(current_plan.tasks)}", task_id=task.task_id, description=task.description[:100] + "..." if len(task.description) > 100 else task.description) # Log preview
                state_tracker.start_task(current_plan.plan_id, task.task_id, task.description, task.dependencies)
                task.status = TaskStatus.IN_PROGRESS

                # 4a. Analyze context for the task
                analyzed_context = await reasoner.analyze_context(task, project_context=project_ctx)
                logger.debug("Context analyzed for task", task_id=task.task_id, analyzed_context_keys=list(analyzed_context.keys()))

                # 4b. Reason about the task
                reasoning_output = await reasoner.reason_about_task(task, analyzed_context)
                logger.debug("Reasoning complete for task", task_id=task.task_id, reasoning_output_keys=list(reasoning_output.keys()))

                # 4c. Make a decision
                decision = reasoner.make_decision(reasoning_output)
                logger.info("Decision made for task", task_id=task.task_id, decision_action=decision.get('action'), confidence=decision.get('confidence'))
                state_tracker.update_task_reasoning(current_plan.plan_id, task.task_id, decision)
                task.reasoning_log.append(f"InitialReasoning: {decision}") # Store decision summary

                # 4d. Potentially trigger research based on decision or task type
                task_knowledge_chunks: List[KnowledgeChunk] = []
                needs_research_keywords = ["research", "find information", "investigate", "what is", "how to", "learn about", "discover"]
                trigger_research = any(kw in task.description.lower() for kw in needs_research_keywords) or \
                                   (decision.get('action') == "NEEDS_CLARIFICATION" and decision.get('confidence', 1.0) < 0.6)

                if trigger_research:
                    logger.info("Task flagged for research, invoking IntelligentResearchAssistant.", task_id=task.task_id)
                    project_summary_for_research = project_ctx.project_description if project_ctx else None

                    current_research_query = ResearchQuery(
                        original_task_description=task.description,
                        project_context_summary=project_summary_for_research
                        # keywords will be extracted by the research_assistant
                    )

                    task_knowledge_chunks = await research_assistant.research_for_task(current_research_query)

                    if task_knowledge_chunks:
                        logger.info("Research completed for task, knowledge chunks obtained.",
                                    task_id=task.task_id, num_chunks=len(task_knowledge_chunks))
                        task.reasoning_log.append(f"ResearchSummary: Found {len(task_knowledge_chunks)} knowledge chunks. First chunk: {task_knowledge_chunks[0].content[:100]}..." if task_knowledge_chunks else "No chunks.")
                        # Here, one might re-evaluate the task or decision based on new knowledge.
                        # For this simulation, we'll just log it and potentially adjust the task message.
                    else:
                        logger.info("Research completed for task, but no knowledge chunks were synthesized.", task_id=task.task_id)
                        task.reasoning_log.append("ResearchSummary: No knowledge chunks synthesized.")

                # 4e. Simulate execution based on decision (and potentially research)
                if decision.get('action') == "PROCEED":
                    time.sleep(0.1) # Simulate work
                    sim_message = "Task simulated successfully."
                    if task_knowledge_chunks:
                        sim_message += f" (Research found {len(task_knowledge_chunks)} knowledge chunks that could be used)."
                    task_result = TaskResult(task_id=task.task_id, status=TaskStatus.COMPLETED, message=sim_message)
                    task.status = TaskStatus.COMPLETED
                    task.output = "Simulated output data. " + (f"Knowledge chunks: {[k.content for k in task_knowledge_chunks]}" if task_knowledge_chunks else "")
                    state_tracker.complete_task(current_plan.plan_id, task.task_id, task.status.value, task_result.message, output_summary="Simulated output produced.")
                else: # NEEDS_CLARIFICATION
                    clarification_message = decision.get('details', "Task requires clarification.")
                    if task_knowledge_chunks: # If research was done due to clarification, it might alter the message
                        clarification_message += f" Research attempted and found {len(task_knowledge_chunks)} chunks; review them to refine the task."

                    task_result = TaskResult(task_id=task.task_id, status=TaskStatus.CLARIFICATION_NEEDED, message=clarification_message)
                    task.status = TaskStatus.CLARIFICATION_NEEDED
                    all_tasks_successful = False
                    state_tracker.complete_task(current_plan.plan_id, task.task_id, task.status.value, task_result.message)


            # 5. Update overall plan status
            if all_tasks_successful:
                current_plan.overall_status = TaskStatus.COMPLETED
                logger.info("All tasks processed successfully.", plan_id=current_plan.plan_id)
            else:
                current_plan.overall_status = TaskStatus.CLARIFICATION_NEEDED # Or FAILED, depending on desired semantics
                logger.warn("Some tasks require clarification. Plan needs review.", plan_id=current_plan.plan_id)

        total_execution_time = time.monotonic() - start_time_total
        if current_plan: # Ensure current_plan is not None before accessing attributes
            state_tracker.complete_plan(current_plan.plan_id, current_plan.overall_status.value, total_execution_time)
            logger.info("Workflow simulation finished.", plan_id=current_plan.plan_id, overall_status=current_plan.overall_status.value, duration_seconds=round(total_execution_time,2))
        else:
            logger.error("Execution plan was not created. Cannot complete plan normally.")
            # Log a generic plan failure if no plan was ever created.
            state_tracker._log_event("plan_creation_failed", plan_id="unknown", task_id=None, details={"reason": "Execution plan object is None before completion."})


    except Exception as e:
        logger.error("Critical error during workflow simulation", error=str(e), exc_info=True)
        if current_plan and current_plan.plan_id:
             state_tracker.complete_plan(current_plan.plan_id, TaskStatus.FAILED.value, time.monotonic() - start_time_total)
        else: # No plan_id to log against if error was very early
             state_tracker._log_event("plan_failed_early", None, None, {"error": str(e)})


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
