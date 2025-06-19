# Role-Based Agent System in OpenHands

## Introduction

The Role-Based Agent System in OpenHands provides a structured way to create and manage specialized AI agents that can collaborate on complex tasks. Inspired by frameworks like CrewAI and MetaGPT, this system allows developers to define distinct roles for agents, assign them specific goals and tools, and orchestrate their interactions through a `CrewManager`. This approach promotes modularity, clarity, and extensibility in building sophisticated multi-agent applications.

## Core Components & Data Models

The system is built around several key Pydantic models and classes:

*   **`RoleDefinition`** (`src.core.agent_structures.RoleDefinition`):
    *   Defines the blueprint for a type of agent.
    *   Fields:
        *   `name: str`: Unique name for the role (e.g., "SoftwarePlanner", "PythonDeveloper").
        *   `description: str`: A brief description of the role.
        *   `goals: List[str]`: Key objectives for this role.
        *   `responsibilities: List[str]`: Specific duties the role performs.
        *   `tools: List[str]`: List of tool names this role is permitted to use. These names should match tools registered in the `ToolsRegistry`.
        *   `input_schema: Optional[Dict[str, Any]]`: (Future use) JSON schema defining expected input for tasks handled by this role.
        *   `output_schema: Optional[Dict[str, Any]]`: (Future use) JSON schema defining the structure of output/artifacts produced by this role.
    *   *Example (conceptual YAML representation for future config files):*
      ```yaml
      - name: PythonDeveloper
        description: Writes, tests, and debugs Python code.
        goals: ["Implement features as per specification", "Ensure code quality"]
        responsibilities: ["Write Python functions/classes", "Write unit tests", "Debug code"]
        tools: ["file_write", "python_interpreter", "code_linter"]
      ```

*   **`AgentConfig`** (`src.core.agent_structures.AgentConfig`):
    *   Configuration for a specific agent instance.
    *   Fields:
        *   `agent_id: str`: Unique identifier for the agent instance.
        *   `role_name: str`: The name of the `RoleDefinition` this agent embodies.
        *   `llm_config: Optional[Dict[str, Any]]`: Specific LLM settings (e.g., model, temperature) for this agent, overriding global or `LLMAggregator` defaults.
        *   `backstory: Optional[str]`: A narrative to help the LLM better embody the role (inspired by CrewAI).
        *   `allow_delegation: bool = False`: (Future use) Whether this agent can delegate tasks.
        *   `max_iterations: int = 10`: (Future use) Maximum iterations for an agent on a task.

*   **`TaskContext`** (`src.core.agent_structures.TaskContext`):
    *   Provides all necessary information to an agent when it executes a task.
    *   Fields:
        *   `project_context: Optional[ProjectContext]`: Overall context of the project the agent is working on. (Type from `src.core.planning_structures`)
        *   `current_task: Task`: The specific `Task` object (from `src.core.planning_structures`) the agent needs to execute.
        *   `execution_plan: Optional[ExecutionPlan]`: The broader plan this task is part of. (Type from `src.core.planning_structures`)
        *   `shared_memory: Dict[str, Any]`: A dictionary for agents within a crew/process to share intermediate results or state.
        *   `history: List[Dict[str, Any]]`: A log of actions taken or messages exchanged relevant to the current task or plan execution.

*   **`TaskResult`** (`src.core.agent_structures.TaskResult`):
    *   The standardized structure for returning the outcome of an agent's task execution.
    *   Fields:
        *   `task_id: str`: The ID of the task that was executed.
        *   `status: TaskStatus`: The outcome status (e.g., `COMPLETED`, `FAILED`). (Type from `src.core.planning_structures.TaskStatus`).
        *   `output: Optional[Any]`: The primary result or artifact produced by the agent for the task.
        *   `error_message: Optional[str]`: Details if the task failed.
        *   `tokens_used: Optional[int]`: LLM tokens consumed.
        *   `cost: Optional[float]`: Cost associated with task execution.

*   **`AbstractBaseAgent`** (`src.core.base_agent.AbstractBaseAgent`):
    *   An abstract base class that all concrete agent role implementations must inherit from.
    *   Key abstract method: `async def execute_task(self, task: Task, context: TaskContext) -> TaskResult`. (Note: `task` argument is `planning_structures.Task`, `context.current_task` is also `planning_structures.Task`).
    *   Provides helpers: `async def _use_llm(self, messages: List[ChatMessage], ...)` for LLM interaction and `async def _use_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolOutput` for using registered tools.
    *   Initialized with `AgentConfig`, `LLMAggregator`, and `ToolsRegistry`.

*   **`ToolsRegistry`** (`src.core.tools_registry.ToolsRegistry`):
    *   A registry for managing and invoking tools that agents can use. Each role can be specified to have access to certain tools. See "Working with Tools" section for more details.

*   **`CrewManager`** (`src.core.crew_manager.CrewManager`):
    *   Orchestrates a "crew" of agents.
    *   Instantiates agents based on `AgentConfig` and a mapping of role names to agent classes.
    *   Processes an `ExecutionPlan` by routing its `Task`s to agents based on the `task.required_role` field.
    *   Currently supports sequential execution of tasks.

## Working with Tools

Tools are essential for agents to interact with the external environment, perform specific actions (like file I/O), or access specialized data sources.

### `ToolInterface` and `ToolOutput`

*   **`ToolInterface`** (`src.core.tool_interface.ToolInterface`):
    *   This is an Abstract Base Class (ABC) that all tools must implement.
    *   It defines a standard contract for tools, ensuring they can be consistently managed and invoked.
    *   **Key Properties:**
        *   `name: str`: A unique name for the tool (e.g., "file_read", "web_search").
        *   `description: str`: A human-readable description of what the tool does, its purpose, and perhaps basic usage.
        *   `input_schema: Type[BaseModel]`: A Pydantic `BaseModel` class that defines the expected input parameters for the tool. This allows for automatic validation of inputs.
        *   `output_schema: Type[ToolOutput]`: A Pydantic `BaseModel` class (typically `ToolOutput` itself or a subclass) that defines the structure of the tool's output.
    *   **Key Method:**
        *   `async def execute(self, inputs: BaseModel) -> ToolOutput`: The core method that performs the tool's action. It receives a validated Pydantic model instance (matching `input_schema`) and must return a `ToolOutput` object.

*   **`ToolOutput`** (`src.core.tool_interface.ToolOutput`):
    *   A Pydantic `BaseModel` used as the standard return type for all tool executions.
    *   **Fields:**
        *   `output: Optional[Any]`: The actual result of the tool's execution if successful (e.g., file content, search results).
        *   `error: Optional[str]`: A string describing an error if the tool execution failed.
        *   `status_code: int`: An HTTP-like status code indicating the outcome (e.g., `200` for success, `400` for bad input, `404` for not found, `500` for internal tool error).
        *   `succeeded: bool` (property): A helper property that returns `True` if `error` is `None` and `status_code` is in the 2xx range.

### `ToolsRegistry`

*   **`ToolsRegistry`** (`src.core.tools_registry.ToolsRegistry`):
    *   Manages the lifecycle and invocation of all available tools within the system.
    *   **Key Methods:**
        *   `register_tool(self, tool: ToolInterface)`: Adds an instance of a tool (which must implement `ToolInterface`) to the registry, making it available for agents.
        *   `unregister_tool(self, tool_name: str)`: Removes a tool from the registry.
        *   `get_tool(self, tool_name: str) -> Optional[ToolInterface]`: Retrieves a registered tool instance by its name.
        *   `async def invoke_tool(self, tool_name: str, inputs_dict: Dict[str, Any]) -> ToolOutput`:
            *   This is the primary method agents use (via `AbstractBaseAgent._use_tool`) to execute a tool.
            *   It finds the tool by `tool_name`.
            *   It validates the provided `inputs_dict` against the tool's `input_schema`. If validation fails, it returns a `ToolOutput` with an error and status code `400`.
            *   It calls the tool's `execute` method with the validated input model.
            *   It catches exceptions during tool execution and wraps them in a `ToolOutput` with an error message and status code `500`.

### Creating a New Tool

To add new functionality for agents, you can create custom tools.

1.  **Define Input Schema:** Create a Pydantic `BaseModel` for your tool's input parameters.
2.  **Implement `ToolInterface`:** Create a class that inherits from `ToolInterface` and implements all its abstract properties and the `execute` method.
3.  **Register the Tool:** Instantiate your tool and register it with an instance of `ToolsRegistry`.

**Example Template:**
```python
# In, e.g., src/core/tools/my_custom_tool.py
from pydantic import BaseModel, Field
from src.core.tool_interface import ToolInterface, ToolOutput
from typing import Type, Any
import structlog

logger = structlog.get_logger(__name__)

class MyToolInput(BaseModel):
    my_param: str = Field(description="An example string parameter for the custom tool.")
    optional_param: int = Field(default=0, description="An optional integer parameter.")

# Example of a more specific output structure if needed, though often ToolOutput is sufficient
# class MyToolOutputPayload(BaseModel):
#     processed_data: str
#     value_calculated: float

class MyCustomTool(ToolInterface):
    @property
    def name(self) -> str:
        return "my_custom_tool"

    @property
    def description(self) -> str:
        return "This is a description of my custom tool. It processes a string and an optional integer."

    @property
    def input_schema(self) -> Type[BaseModel]:
        return MyToolInput

    @property
    def output_schema(self) -> Type[ToolOutput]: # Can also be a custom ToolOutput subclass
        return ToolOutput

    async def execute(self, inputs: MyToolInput) -> ToolOutput:
        # 'inputs' is already a validated MyToolInput instance due to ToolsRegistry
        logger.info(f"Executing {self.name} with param: '{inputs.my_param}' and optional_param: {inputs.optional_param}")
        try:
            # Your tool's core logic here
            result_data = f"Successfully processed: {inputs.my_param.upper()} with optional value {inputs.optional_param}"

            # If you had a specific payload model like MyToolOutputPayload:
            # payload = MyToolOutputPayload(processed_data=result_data, value_calculated=len(inputs.my_param) + inputs.optional_param)
            # return ToolOutput(output=payload, status_code=200)

            return ToolOutput(output=result_data, status_code=200)
        except Exception as e:
            logger.error(f"Error in {self.name}", error=str(e), exc_info=True)
            return ToolOutput(error=f"Execution failed in {self.name}: {str(e)}", status_code=500)
```

**Registering the Custom Tool:**
```python
# Example of registering the tool (typically done during application setup)
# from src.core.tools_registry import ToolsRegistry
# from src.core.tools.my_custom_tool import MyCustomTool # Assuming file location

# tools_registry_instance = ToolsRegistry()
# my_tool = MyCustomTool()
# tools_registry_instance.register_tool(my_tool)

# Now "my_custom_tool" can be invoked via tools_registry_instance.invoke_tool()
```

### Agents Using Tools

Agents interact with tools via the `_use_tool` helper method provided by `AbstractBaseAgent`.

*   **`AbstractBaseAgent._use_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolOutput`**:
    *   This asynchronous method handles the call to `ToolsRegistry.invoke_tool`.
    *   It simplifies tool usage for agent developers.
*   **Permissions:** An agent's `RoleDefinition` includes a `tools: List[str]` field. This list is intended to specify which tools (by name) an agent in that role is permitted to use. While the actual permission enforcement logic in `_use_tool` is a future enhancement, the design is for `RoleDefinition.tools` to be the source of truth for tool access rights.
*   **Error Handling:** Agents should always check the `ToolOutput.error` and `ToolOutput.status_code` (or `ToolOutput.succeeded` property) to handle potential failures in tool execution gracefully.

**Conceptual Example within an Agent's `execute_task`:**
```python
# Inside an agent's execute_task method:
# async def execute_task(self, task: Task, context: TaskContext) -> TaskResult:
#     # ... other agent logic ...

#     # Example: Using "my_custom_tool"
#     tool_inputs = {"my_param": "hello world", "optional_param": 10}
#     # Assume self.agent_config.tools (derived from RoleDefinition) includes "my_custom_tool"
#     # A real permission check would happen in _use_tool or be pre-checked.

#     tool_result = await self._use_tool("my_custom_tool", tool_inputs)

#     if not tool_result.succeeded: # Check the succeeded property or error/status_code
#         error_message = f"Tool 'my_custom_tool' failed: {tool_result.error} (Status: {tool_result.status_code})"
#         logger.error(error_message)
#         # Handle error, perhaps return a FAILED TaskResult or try a different approach
#         return TaskResult(task_id=task.task_id, status=TaskStatus.FAILED, error_message=error_message)

#     successful_tool_output = tool_result.output
#     logger.info(f"Tool 'my_custom_tool' output: {successful_tool_output}")

#     # ... proceed with successful_tool_output ...
#     # return TaskResult(task_id=task.task_id, status=TaskStatus.COMPLETED, output=successful_tool_output)
```

## Workflow Overview

1.  **Define Roles & Tools:** `RoleDefinition` objects are created, including the list of allowed `tools` for each role. Tools implementing `ToolInterface` are defined and registered with a `ToolsRegistry` instance.
2.  **Configure Agents:** `AgentConfig` objects are created, assigning specific roles to agent instances and providing any unique configurations.
3.  **Initialize CrewManager:** A `CrewManager` is instantiated with a list of `AgentConfig`s, a shared `LLMAggregator`, the populated `ToolsRegistry`, and the defined `RoleDefinition`s. The `CrewManager` creates the actual agent instances.
4.  **Obtain ExecutionPlan:** An `ExecutionPlan` (a list of `Task` objects with dependencies, potentially generated by a `PlannerAgent`) is provided to the `CrewManager`. Each `Task` should have its `required_role` field set.
5.  **Process Plan:** The `CrewManager` iterates through the tasks in the plan. For each task:
    a. It determines the appropriate agent by looking up `task.required_role` in its `agents_by_role` mapping. If the role is not specified on the task, or if no agent is configured for that role, the task is marked as FAILED.
    b. It prepares a `TaskContext`.
    c. It calls the `execute_task` method of the selected agent.
6.  **Agent Executes Task:** The agent performs its role-specific logic (often following an internal SOP). If it needs to use a tool:
    a. It calls `await self._use_tool("tool_name", {...inputs...})`.
    b. It checks the returned `ToolOutput` for errors and processes the `output` if successful.
    c. It may use `self._use_llm` for AI capabilities, potentially using tool outputs as part of the LLM prompt.
7.  **Return Result:** The agent returns a `TaskResult` summarizing its execution outcome for the task.
8.  **Update Plan:** The `CrewManager` updates the task's status and output (and potentially error messages) in the `ExecutionPlan` and proceeds to the next task. If a task fails, the `CrewManager` currently stops processing the plan and marks the overall plan as FAILED.

## Defining and Using Roles & Agents (Programmatic Setup)

Currently, roles and agents are defined programmatically. Future enhancements may allow loading from YAML configurations.

```python
from src.core.agent_structures import RoleDefinition, AgentConfig
from src.core.planning_structures import ProjectContext, Task, ExecutionPlan, TaskStatus
from src.core.crew_manager import CrewManager
from src.core.aggregator import LLMAggregator
from src.core.tools_registry import ToolsRegistry
# Concrete Agent Classes (ensure they are imported for AGENT_CLASS_MAP in CrewManager)
# from src.core.agents.planner_agent import PlannerAgent
# from src.core.agents.developer_agent import DeveloperAgent
# from src.core.agents.researcher_agent import ResearcherAgent
# from src.core.agents.qa_agent import QAAgent # Added QAAgent
# from src.core.tools.file_system_tools import ReadFileTool, WriteFileTool # Example tools

# 1. Setup LLMAggregator and ToolsRegistry (placeholders for actual setup)
# llm_aggregator = LLMAggregator(...) # Requires actual setup
# tools_registry = ToolsRegistry()
# # Register any tools that will be used
# tools_registry.register_tool(ReadFileTool())
# tools_registry.register_tool(WriteFileTool())
# Note: Actual instantiation of LLMAggregator will require its dependencies.

# 2. Define Roles
planner_role_def = RoleDefinition(
    name="SoftwarePlanner",
    description="Analyzes requirements and creates detailed execution plans.",
    goals=["Decompose complex tasks"],
    responsibilities=["Parse user intent", "Generate task lists"],
    tools=[]
)
developer_role_def = RoleDefinition(
    name="PythonDeveloper",
    description="Writes, tests, and debugs Python code.",
    goals=["Implement features", "Ensure code quality"],
    responsibilities=["Write Python code", "Write unit tests", "Read/Write files as needed"],
    tools=["file_read", "file_write"]
)
qa_role_def = RoleDefinition(
    name="QualityAssuranceAgent",
    description="Reviews content for quality and adherence to requirements.",
    goals=["Ensure high quality", "Identify defects"],
    responsibilities=["Review code/text", "Validate outputs"],
    tools=[] # Primarily uses LLM for review
)
role_definitions_map = {
    planner_role_def.name: planner_role_def,
    developer_role_def.name: developer_role_def,
    qa_role_def.name: qa_role_def, # Added QA role
}

# 3. Configure Agents
planner_agent_config = AgentConfig(agent_id="planner_alpha", role_name="SoftwarePlanner")
developer_agent_config = AgentConfig(agent_id="developer_beta", role_name="PythonDeveloper")
qa_agent_config = AgentConfig(agent_id="qa_gamma", role_name="QualityAssuranceAgent") # Added QA agent config

agent_configs = [planner_agent_config, developer_agent_config, qa_agent_config]

# 4. Initialize CrewManager
# Ensure AGENT_CLASS_MAP in crew_manager.py includes:
# "QualityAssuranceAgent": QAAgent
# crew_manager = CrewManager(
#     agent_configs=agent_configs,
#     llm_aggregator=llm_aggregator,
#     tools_registry=tools_registry,
#     role_definitions=role_definitions_map
# )

# 5. Create an ExecutionPlan (e.g., manually or from a PlannerAgent)
# For this example, assume a PlannerAgent (or an initial task for it) would produce this:
# plan = ExecutionPlan(
#     user_instruction="Create a CLI tool that reads a file, processes its content, and writes to another file, then QA the output.",
#     parsed_intent={"goal": "Create a CLI tool with file I/O and QA"},
#     tasks=[
#         Task(task_id="task_1", description="Plan the CLI tool features", required_role="SoftwarePlanner", dependencies=[]),
#         Task(task_id="task_2", description="Implement the file reading part: read file 'input.txt'", required_role="PythonDeveloper", dependencies=["task_1"]),
#         Task(task_id="task_3", description="Implement the core logic for processing content.", required_role="PythonDeveloper", dependencies=["task_2"]),
#         Task(task_id="task_4", description="Implement file writing part: write file 'output.txt' with processed content.", required_role="PythonDeveloper", dependencies=["task_3"]),
#         Task(task_id="task_5", description="QA the generated 'output.txt' file.", required_role="QualityAssuranceAgent", raw_instruction=json.dumps({"content_to_qa": "output_from_task_4", "qa_criteria": "Ensure output is valid text."}), dependencies=["task_4"])
#     ]
# )
# project_ctx = ProjectContext(project_id="example_project", project_name="CLI Tool Example")

# 6. Process the plan (conceptual - actual call would be in an async context)
# async def run_example():
#    # (LLMAggregator and ToolsRegistry need proper async setup if their methods are async)
#    # llm_aggregator_instance = ...
#    # tools_registry_instance = ... (with tools registered)
#    # crew_manager = CrewManager(
#    #    agent_configs=agent_configs,
#    #    llm_aggregator=llm_aggregator_instance,
#    #    tools_registry=tools_registry_instance,
#    #    role_definitions=role_definitions_map
#    # )
#    # updated_plan = await crew_manager.process_execution_plan(plan, project_ctx)
#    # print(f"Plan Status: {updated_plan.overall_status}")
#    # for task_item in updated_plan.tasks:
#    #     print(f"Task {task_item.task_id} ({task_item.description[:30]}...): {task_item.status} - Output: {type(task_item.output)}")
# pass # Placeholder for async execution example
```

## Implementing a New Agent Role

To create a new agent role:
1.  Define its `RoleDefinition` (as shown above), including the `tools` list if it uses any.
2.  Create a new Python class that inherits from `AbstractBaseAgent` (e.g., in `src/core/agents/`).
3.  Implement the `async def execute_task(self, task: Task, context: TaskContext) -> TaskResult:` method. This method orchestrates the agent's Standard Operating Procedure (SOP). The SOP is typically broken down into several private helper methods (e.g., `_sop_step1_analyze(...)`, `_sop_step2_process(...)`, etc.) to structure the agent's internal logic.
    *   The `DeveloperAgent` follows an internal SOP which typically includes: 1. Analyzing Requirements, 2. Reading Initial Files (if specified), 3. Generating Code, 4. Performing Self-Critique on Code, 5. Writing Output Files (if specified).
    *   The `ResearcherAgent` follows an internal SOP which typically includes: 1. Clarifying Research Goal, 2. Executing Research Cycle (using `IntelligentResearchAssistant`), 3. Formatting the Report, 4. Saving the Report to a File (if specified).
    *   The `QAAgent` reviews content against criteria. Its input is typically from `task.raw_instruction` (parsed as JSON for `content_to_qa` and `qa_criteria`, with a fallback to `task.description` for `content_to_qa`). It outputs a JSON structure with assessment, issues, suggestions, and confidence.
4.  Add your new agent class to the `AGENT_CLASS_MAP` dictionary in `src/core/crew_manager.py` using the `RoleDefinition.name` as the key.

**Example Template:**
```python
from src.core.base_agent import AbstractBaseAgent
from src.core.agent_structures import AgentConfig, TaskContext, TaskResult
from src.core.planning_structures import Task, TaskStatus # Task from planning_structures, TaskStatus for result
from src.core.aggregator import LLMAggregator
from src.core.tools_registry import ToolsRegistry
from src.core.tool_interface import ToolOutput # For checking tool results
from src.models import ChatMessage # For _use_llm example
import structlog
from typing import Optional, Any

logger = structlog.get_logger(__name__)

class MyCustomAgentWithTools(AbstractBaseAgent):
    def __init__(self,
                 agent_config: AgentConfig,
                 llm_aggregator: LLMAggregator,
                 tools_registry: Optional[ToolsRegistry] = None):
        super().__init__(agent_config, llm_aggregator, tools_registry)
        logger.info(f"MyCustomAgentWithTools '{self.agent_config.agent_id}' for role '{self.agent_config.role_name}' initialized.")
        # Initialize any role-specific components here

    async def execute_task(self, task: Task, context: TaskContext) -> TaskResult:
        logger.info(f"MyCustomAgentWithTools '{self.agent_config.agent_id}' executing task: {task.task_id} - {task.description}")

        # Example: Using a tool if mentioned in the task description
        if "use my tool for" in task.description.lower():
            # This is a simplified way to get parameters; real parsing or structured input is better
            param_value = task.description.split("use my tool for")[-1].strip()
            tool_result: ToolOutput = await self._use_tool("my_custom_tool", {"my_param": param_value})

            if not tool_result.succeeded:
                error_msg = f"Tool 'my_custom_tool' failed: {tool_result.error} (Code: {tool_result.status_code})"
                logger.error(error_msg)
                return TaskResult(task_id=task.task_id, status=TaskStatus.FAILED, error_message=error_msg)

            processed_data_from_tool = tool_result.output
            # Now use processed_data_from_tool, perhaps with an LLM
            final_llm_prompt = f"Based on tool output: '{processed_data_from_tool}', complete the task: {task.description}"
        else:
            final_llm_prompt = f"Complete the task: {task.description}"

        # LLM call
        try:
            response = await self._use_llm([ChatMessage(role="user", content=final_llm_prompt)])
            llm_output = response.choices[0].message.content if response.choices and response.choices[0].message else "No LLM output"
        except Exception as e:
            logger.error("LLM call failed", error=str(e), exc_info=True)
            return TaskResult(task_id=task.task_id, status=TaskStatus.FAILED, error_message=f"LLM error: {e}")

        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output={"llm_summary": llm_output}
        )
```

## Future Enhancements
- Loading `RoleDefinition` and `AgentConfig` from YAML files.
- More sophisticated task routing within `CrewManager` (e.g., agent availability, load balancing, skill-based routing beyond just role).
- Support for different process models (e.g., hierarchical task delegation, parallel execution of independent tasks).
- Full implementation of permission checks in `AbstractBaseAgent._use_tool` based on `RoleDefinition.tools`.
- Advanced inter-agent communication mechanisms and richer `shared_memory` utilization patterns.
- Integration with `StateTracker` for more comprehensive logging of agent actions and state changes.
