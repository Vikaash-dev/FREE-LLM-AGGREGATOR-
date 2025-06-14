from typing import Dict, Any, Optional # Added Optional
import structlog
import json # For parsing LLM response if it's a JSON string

from .planning_structures import ExecutionPlan, Task, TaskStatus, ProjectContext
# We need access to LLMAggregator. This might require some refactoring
# or specific instantiation logic later. For now, let's assume it can be passed.
from .aggregator import LLMAggregator
from ..models import ChatCompletionRequest, Message


logger = structlog.get_logger(__name__)

class DevikaInspiredPlanner:
    '''
    A planner inspired by Devika AI's capabilities, responsible for
    decomposing complex tasks and creating execution plans.
    '''

    def __init__(self, llm_aggregator: LLMAggregator):
        '''
        Initializes the DevikaInspiredPlanner.

        Args:
            llm_aggregator: An instance of LLMAggregator to interact with LLMs.
        '''
        self.llm_aggregator = llm_aggregator
        logger.info("DevikaInspiredPlanner initialized.")

    async def parse_user_intent(self, instruction: str, context: Optional[ProjectContext] = None) -> Dict[str, Any]:
        '''
        Parses the user's textual instruction to extract a structured intent.

        Args:
            instruction: The raw textual instruction from the user.
            context: Optional project context.

        Returns:
            A dictionary representing the structured intent.
            Example: {'goal': '...', 'entities': [], 'constraints': [], 'raw_instruction': instruction }
        '''
        logger.info("Parsing user intent", raw_instruction=instruction, project_context=context.project_id if context else None)

        prompt_template = f"""
        You are an AI assistant helping to understand a user's request.
        Analyze the following user instruction and extract the main goal,
        any key entities mentioned, and any constraints or specific requirements.

        User Instruction: "{instruction}"

        Provide the output as a JSON object with the following keys:
        - "goal": (string) A concise statement of the user's primary objective.
        - "entities": (list of strings) Key nouns or objects relevant to the goal.
        - "constraints": (list of strings) Any limitations, conditions, or specific methods mentioned.
        - "raw_instruction": (string) The original user instruction.

        Example:
        User Instruction: "Create a Python script to parse a CSV file named 'data.csv' and print the first 10 rows. The script should be efficient."
        Output:
        {{
            "goal": "Create a Python script to parse 'data.csv' and print its first 10 rows",
            "entities": ["Python script", "CSV file", "data.csv", "10 rows"],
            "constraints": ["script should be efficient"],
            "raw_instruction": "Create a Python script to parse a CSV file named 'data.csv' and print the first 10 rows. The script should be efficient."
        }}

        Now, analyze the provided User Instruction.
        User Instruction: "{instruction}"
        Output:
        """

        messages = [
            Message(role="system", content="You are an expert in understanding and structuring user requests into actionable components."),
            Message(role="user", content=prompt_template)
        ]

        # Assuming 'auto' model selection or a default model configured in LLMAggregator
        # and that the response will be a JSON string.
        request = ChatCompletionRequest(messages=messages, model="auto", temperature=0.2) # type: ignore

        try:
            logger.debug("Sending intent parsing request to LLM", instruction_length=len(instruction))
            response = await self.llm_aggregator.chat_completion(request)

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                content_str = response.choices[0].message.content.strip()
                logger.debug("Received LLM response for intent parsing", response_content_length=len(content_str))

                # Basic cleanup if LLM wraps with ```json ... ```
                if content_str.startswith("```json"):
                    content_str = content_str[len("```json"):]
                if content_str.endswith("```"):
                    content_str = content_str[:-len("```")]

                import json
                try:
                    parsed_response = json.loads(content_str)
                    # Ensure raw_instruction is always present
                    if "raw_instruction" not in parsed_response:
                        parsed_response["raw_instruction"] = instruction
                    logger.info("Successfully parsed user intent", goal=parsed_response.get("goal"))
                    return parsed_response
                except json.JSONDecodeError as e:
                    logger.error("Failed to decode JSON from LLM response for intent parsing",
                                 error=str(e),
                                 raw_response_snippet=content_str[:200]) # Log snippet to avoid huge logs
                    # Fallback in case of JSON error
                    return {
                        "goal": instruction, # Use raw instruction as goal
                        "entities": [],
                        "constraints": ["Failed to parse structured intent from LLM."],
                        "raw_instruction": instruction
                    }
            else:
                logger.warn("LLM response for intent parsing was empty or malformed.", llm_response_obj=response.model_dump_json() if response else None)
                return {
                    "goal": instruction,
                    "entities": [],
                    "constraints": ["LLM response was empty or malformed."],
                    "raw_instruction": instruction
                }

        except Exception as e:
            logger.error("Error during LLM call for intent parsing", error=str(e), exc_info=True)
            return {
                "goal": instruction,
                "entities": [],
                "constraints": [f"An error occurred: {str(e)}"],
                "raw_instruction": instruction
            }

    async def decompose_complex_task(self, parsed_intent: Dict[str, Any], context: Optional[ProjectContext] = None) -> ExecutionPlan:
        '''
        Decomposes a parsed user intent into a sequence of sub-tasks.

        Args:
            parsed_intent: The structured output from parse_user_intent.
            context: Optional project context.

        Returns:
            An ExecutionPlan object containing the list of decomposed tasks.
        '''
        goal = parsed_intent.get("goal", parsed_intent.get("raw_instruction", "No goal specified"))
        raw_instruction = parsed_intent.get("raw_instruction", goal)
        entities = parsed_intent.get("entities", [])
        constraints = parsed_intent.get("constraints", [])

        logger.info("Decomposing complex task", goal=goal, entities=entities, constraints=constraints, project_context=context.project_id if context else None)

        # Prepare a detailed prompt for the LLM
        # We need to instruct the LLM to output a list of tasks in a specific JSON format.
        prompt_template = f"""
        You are an AI assistant responsible for breaking down a complex user goal into a sequence of smaller, actionable sub-tasks.
        The user's overall goal is: "{goal}"
        Key entities identified: {', '.join(entities) if entities else 'None'}
        Specific constraints or requirements: {', '.join(constraints) if constraints else 'None'}
        The original user instruction was: "{raw_instruction}"

        Please decompose this goal into a list of sub-tasks. Each sub-task should be a clear, concise action.
        For each sub-task, provide:
        1.  "task_id": A unique placeholder string like "task_N" (e.g., "task_1", "task_2"). You will generate these.
        2.  "description": A string describing the action to be performed for this sub-task.
        3.  "dependencies": A list of "task_id" strings that this sub-task depends on. For now, assume tasks are mostly sequential unless simple parallelism is obvious. If sequential, task_N would depend on task_(N-1). For the first task, dependencies should be an empty list.

        Example of desired output format (a JSON list of objects):
        [
            {{
                "task_id": "task_1",
                "description": "Set up the project environment for a Python application.",
                "dependencies": []
            }},
            {{
                "task_id": "task_2",
                "description": "Install necessary libraries like pandas and numpy.",
                "dependencies": ["task_1"]
            }},
            {{
                "task_id": "task_3",
                "description": "Write the main script to load 'data.csv'.",
                "dependencies": ["task_2"]
            }}
        ]

        Based on the goal, entities, and constraints, provide the list of sub-tasks in the JSON format described above.
        Ensure the task descriptions are actionable and clear.
        Ensure dependencies correctly reflect a logical sequence of operations.
        Output ONLY the JSON list of tasks. Do not include any other explanatory text before or after the JSON.
        """

        messages = [
            Message(role="system", content="You are an expert task decomposition AI. Your output must be a valid JSON list of task objects as specified."),
            Message(role="user", content=prompt_template)
        ]

        request = ChatCompletionRequest(messages=messages, model="auto", temperature=0.3) # type: ignore # Slightly higher temp for creative task breakdown

        decomposed_tasks_json_str: Optional[str] = None # Initialize with Optional[str]
        try:
            logger.debug("Sending task decomposition request to LLM", goal_length=len(goal))
            response = await self.llm_aggregator.chat_completion(request)

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                decomposed_tasks_json_str = response.choices[0].message.content.strip()
                logger.debug("Received LLM response for task decomposition", response_content_length=len(decomposed_tasks_json_str))

                # Clean up potential markdown code block fences
                if decomposed_tasks_json_str.startswith("```json"):
                    decomposed_tasks_json_str = decomposed_tasks_json_str[len("```json"):] # Corrected length
                if decomposed_tasks_json_str.endswith("```"):
                    decomposed_tasks_json_str = decomposed_tasks_json_str[:-len("```")] # Corrected length

                # Attempt to parse the JSON
                llm_generated_tasks_data = json.loads(decomposed_tasks_json_str)

                tasks: List[Task] = [] # Ensure tasks is typed
                for i, task_data in enumerate(llm_generated_tasks_data):
                    # Ensure task_id is robust, even if LLM doesn't provide it perfectly
                    # Or, we can generate our own UUIDs here and map LLM's task_ids for dependencies.
                    # For now, trust LLM's task_id for simplicity in this phase.
                    task_obj = Task(
                        # task_id is generated by LLM based on prompt
                        task_id=task_data.get("task_id", f"llm_task_{i+1}"),
                        description=task_data.get("description", "No description provided by LLM."),
                        dependencies=task_data.get("dependencies", []),
                        raw_instruction=goal # The sub-task is derived from this goal
                    )
                    tasks.append(task_obj)

                logger.info("Successfully decomposed task into sub-tasks", num_sub_tasks=len(tasks), goal_preview=goal[:50])
                return ExecutionPlan(
                    user_instruction=raw_instruction,
                    parsed_intent=parsed_intent,
                    tasks=tasks,
                    overall_status=TaskStatus.PENDING if tasks else TaskStatus.COMPLETED # If no tasks, plan is complete
                )

            else:
                logger.warn("LLM response for task decomposition was empty or malformed.", llm_response_obj=response.model_dump_json() if response else None, goal_preview=goal[:50])

        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from LLM response for task decomposition",
                         error=str(e), raw_response_snippet=decomposed_tasks_json_str[:200] if decomposed_tasks_json_str else "None", goal_preview=goal[:50])
        except Exception as e:
            logger.error("Error during LLM call for task decomposition", error=str(e), goal_preview=goal[:50], exc_info=True)

        # Fallback: If LLM fails or output is unusable, create a single task from the goal
        logger.warn("Falling back to creating a single task from the goal.", goal_preview=goal[:50])
        single_task = Task(description=goal, raw_instruction=raw_instruction)
        return ExecutionPlan(
            user_instruction=raw_instruction,
            parsed_intent=parsed_intent,
            tasks=[single_task],
            overall_status=TaskStatus.PENDING
        )
