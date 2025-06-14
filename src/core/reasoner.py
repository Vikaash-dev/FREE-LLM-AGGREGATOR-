from typing import Dict, Any, Optional
import structlog
import json # For parsing LLM responses

from .planning_structures import Task, ProjectContext # Core data structures
from .aggregator import LLMAggregator # For potential LLM calls in future reasoning steps
from ..models import ChatCompletionRequest, Message # For LLMAggregator


logger = structlog.get_logger(__name__)

class ContextualReasoningEngine:
    '''
    Responsible for analyzing the context of a task and reasoning about it
    to provide insights, identify ambiguities, or suggest approaches.
    '''

    def __init__(self, llm_aggregator: LLMAggregator):
        '''
        Initializes the ContextualReasoningEngine.

        Args:
            llm_aggregator: An instance of LLMAggregator, which might be used
                            for more advanced reasoning steps in the future.
        '''
        self.llm_aggregator = llm_aggregator # Stored for future use
        logger.info("ContextualReasoningEngine initialized.")

    async def analyze_context(self, task: Task, project_context: Optional[ProjectContext] = None) -> Dict[str, Any]:
        '''
        Analyzes and gathers relevant context for a given task.

        For Phase 1, this gathers basic information from the task itself
        and the provided project context. More sophisticated context gathering
        (e.g., file system analysis, code analysis) would be future enhancements.

        Args:
            task: The Task object to analyze context for.
            project_context: Optional project context information.

        Returns:
            A dictionary summarizing the gathered context.
        '''
        logger.info("Analyzing context for task", task_id=task.task_id, task_description=task.description_preview(50) if hasattr(task, 'description_preview') else task.description[:50])

        gathered_context: Dict[str, Any] = { # Explicitly type gathered_context
            "task_id": task.task_id,
            "task_description": task.description,
            "task_status": task.status.value if task.status else None, # Access .value for Enum
            "task_dependencies": task.dependencies,
            "task_raw_instruction": task.raw_instruction,
        }

        if project_context:
            gathered_context["project_id"] = project_context.project_id
            gathered_context["project_name"] = project_context.project_name
            gathered_context["project_description"] = project_context.project_description
            gathered_context["project_additional_details"] = project_context.additional_details
            logger.debug("Project context included", task_id=task.task_id, project_id=project_context.project_id)
        else:
            logger.debug("No project context provided for task", task_id=task.task_id)

        # In the future, this could be expanded to:
        # - Read relevant file contents from the project.
        # - Summarize previous related task outputs.
        # - Fetch information from a knowledge base.
        # - Analyze code if the task is about code modification.

        logger.debug("Context analysis complete", task_id=task.task_id, gathered_context_keys=list(gathered_context.keys()))
        return gathered_context

    async def reason_about_task(self, task: Task, analyzed_context: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Uses an LLM to reason about the given task within its analyzed context.
        Aims to identify ambiguities, suggest information needed, or outline a high-level approach.

        Args:
            task: The Task object to reason about.
            analyzed_context: The dictionary of context gathered by analyze_context.

        Returns:
            A dictionary representing the LLM's reasoning output.
            Example: {'reasoning_steps': [...], 'potential_issues': [...], 'next_step_suggestion': '...', 'confidence_score': 0.8}
        '''
        logger.info("Reasoning about task", task_id=task.task_id, task_description=task.description[:50] + "..." if len(task.description) > 50 else task.description)

        # Create a summary of the context for the prompt
        context_summary_parts = []
        for key, value in analyzed_context.items():
            if isinstance(value, list) and len(value) > 3: # Summarize long lists
                context_summary_parts.append(f"- {key}: (list with {len(value)} items, e.g., {value[:2]}...)")
            elif isinstance(value, dict) and len(value) > 3:
                 context_summary_parts.append(f"- {key}: (dict with {len(value)} keys, e.g., {list(value.keys())[:2]}...)")
            elif isinstance(value, str) and len(value) > 200: # Truncate long strings
                 context_summary_parts.append(f"- {key}: '{value[:100]}...' (truncated)")
            else:
                 context_summary_parts.append(f"- {key}: {value}")

        context_for_prompt = "\n".join(context_summary_parts)

        prompt_template = f"""
        You are an AI assistant that helps reason about a specific software development task.
        Your goal is to analyze the task and its context, identify potential issues or ambiguities,
        and suggest a high-level approach or if clarification is needed.

        Task Description: "{task.description}"
        Task Raw Instruction (if any): "{task.raw_instruction}"

        Context:
        {context_for_prompt}

        Please provide your reasoning as a JSON object with the following keys:
        - "reasoning_steps": (list of strings) Your thought process or steps to understand the task.
        - "potential_issues": (list of strings) Any ambiguities, missing information, or potential problems you foresee.
        - "information_needed": (list of strings) Specific information that, if provided, would clarify the task.
        - "suggested_approach": (string) A brief, high-level suggestion on how to tackle this task.
        - "confidence_score": (float, 0.0 to 1.0) Your confidence that the task is clear and actionable with the current information.
        - "requires_clarification": (boolean) True if you believe human clarification is essential before proceeding.

        Example Output:
        {{
            "reasoning_steps": [
                "The task is to 'refactor the authentication module'.",
                "The context mentions the project uses JWT tokens.",
                "Refactoring implies improving structure, performance, or readability without changing external behavior."
            ],
            "potential_issues": [
                "The specific goals of the refactoring are not mentioned (e.g., improve performance, reduce complexity, fix specific bugs?).",
                "No mention of specific metrics to evaluate the success of refactoring."
            ],
            "information_needed": [
                "What are the primary objectives for this refactoring?",
                "Are there any parts of the authentication module that should NOT be changed?"
            ],
            "suggested_approach": "Start by reviewing the existing authentication code, then identify areas for improvement based on common refactoring patterns. Seek clarification on objectives before major changes.",
            "confidence_score": 0.6,
            "requires_clarification": true
        }}

        Analyze the provided task and context, and respond ONLY with the JSON object.
        """

        messages = [
            Message(role="system", content="You are an expert reasoning AI. Your output must be a valid JSON object as specified."),
            Message(role="user", content=prompt_template)
        ]

        request = ChatCompletionRequest(messages=messages, model="auto", temperature=0.4) # type: ignore

        llm_reasoning_json_str: Optional[str] = None
        try:
            logger.debug("Sending task reasoning request to LLM", task_id=task.task_id)
            response = await self.llm_aggregator.chat_completion(request)

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                llm_reasoning_json_str = response.choices[0].message.content.strip()
                logger.debug("Received LLM response for task reasoning", task_id=task.task_id, response_content_length=len(llm_reasoning_json_str))

                if llm_reasoning_json_str.startswith("```json"):
                    llm_reasoning_json_str = llm_reasoning_json_str[len("```json"):]
                if llm_reasoning_json_str.endswith("```"):
                    llm_reasoning_json_str = llm_reasoning_json_str[:-len("```")]

                parsed_response = json.loads(llm_reasoning_json_str)
                logger.info("Successfully parsed LLM reasoning for task", task_id=task.task_id, reasoning_output_keys=list(parsed_response.keys()))
                return parsed_response
            else:
                logger.warn("LLM response for task reasoning was empty or malformed.", task_id=task.task_id, llm_response_obj=response.model_dump_json() if response else None)

        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from LLM response for task reasoning",
                         error=str(e), raw_response_snippet=llm_reasoning_json_str[:200] if llm_reasoning_json_str else "None", task_id=task.task_id)
        except Exception as e:
            logger.error("Error during LLM call for task reasoning", error=str(e), task_id=task.task_id, exc_info=True)

        # Fallback response
        return {
            "reasoning_steps": ["LLM reasoning failed or produced invalid output."],
            "potential_issues": ["Could not perform detailed reasoning."],
            "information_needed": [],
            "suggested_approach": "Proceed with caution or seek manual review.",
            "confidence_score": 0.1,
            "requires_clarification": True # Default to needing clarification on failure
        }

    def make_decision(self, reasoning_output: Dict[str, Any]) -> Dict[str, Any]:
        '''
        Makes a simple decision based on the LLM's reasoning output.
        For Phase 1, this primarily checks if the LLM indicated clarification is needed.

        Args:
            reasoning_output: The structured dictionary from reason_about_task.

        Returns:
            A decision object, e.g., {'action': 'PROCEED'/'NEEDS_CLARIFICATION', 'details': '...', 'confidence': 0.8}
        '''
        logger.info("Making decision based on reasoning output",
                    requires_clarification=reasoning_output.get('requires_clarification'),
                    confidence=reasoning_output.get('confidence_score'))

        confidence = reasoning_output.get('confidence_score', 0.1) # Default to low confidence
        requires_clarification = reasoning_output.get('requires_clarification', True) # Default to needing clarification

        if requires_clarification or confidence < 0.5: # Threshold can be tuned
            action = "NEEDS_CLARIFICATION"
            details = "LLM reasoning suggests the task is unclear or confidence is low. " + \
                      f"Potential issues: {reasoning_output.get('potential_issues', [])}. " + \
                      f"Information needed: {reasoning_output.get('information_needed', [])}."
        else:
            action = "PROCEED"
            details = f"Task seems clear enough to proceed. Suggested approach: {reasoning_output.get('suggested_approach', 'N/A')}"

        decision = {
            "action": action,
            "details": details,
            "confidence": confidence,
            "llm_reasoning_summary": {
                "potential_issues": reasoning_output.get('potential_issues', []),
                "information_needed": reasoning_output.get('information_needed', []),
                "suggested_approach": reasoning_output.get('suggested_approach', 'N/A')
            }
        }
        logger.debug("Decision made", decision_action=action, task_confidence=confidence)
        return decision
