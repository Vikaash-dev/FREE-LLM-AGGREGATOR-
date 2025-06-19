from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import structlog # Added structlog

from src.core.agent_structures import AgentConfig, Task, TaskContext, TaskResult
from src.core.aggregator import LLMAggregator
from src.models import ChatMessage, ChatCompletionResponse, ChatCompletionRequest
# Import ToolsRegistry and ToolOutput for _use_tool method
from src.core.tools_registry import ToolsRegistry
from src.core.tool_interface import ToolOutput

logger = structlog.get_logger(__name__) # Added logger


class AbstractBaseAgent(ABC):
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_aggregator: LLMAggregator,
        tools_registry: Optional[ToolsRegistry] = None, # Changed Any to ToolsRegistry
    ):
        self.agent_config = agent_config
        self.llm_aggregator = llm_aggregator
        self.tools_registry = tools_registry
        # TODO: For permission check in _use_tool, agent needs access to its RoleDefinition.
        # This could be passed in __init__ or fetched from a global role registry.
        # self.role_definition: Optional[RoleDefinition] = None

    @abstractmethod
    async def execute_task(self, task: Task, context: TaskContext) -> TaskResult:
        pass

    async def _use_llm(
        self, messages: List[ChatMessage], specific_llm_config: Optional[Dict] = None
    ) -> ChatCompletionResponse:
        """
        Helper method to interact with the LLM.
        Applies specific LLM configurations if provided.
        """
        llm_config_to_use = self.agent_config.llm_config.copy() if self.agent_config.llm_config else {}
        if specific_llm_config:
            llm_config_to_use.update(specific_llm_config)

        request = ChatCompletionRequest(
            messages=messages,
            model=llm_config_to_use.get("model", "auto"), # Default to "auto" if not in merged config
            temperature=llm_config_to_use.get("temperature"),
            top_p=llm_config_to_use.get("top_p"),
            max_tokens=llm_config_to_use.get("max_tokens"),
            stream=llm_config_to_use.get("stream", False), # Default stream to False
            # Add other parameters as needed from llm_config_to_use
        )
        # Log the request details (optional, can be verbose)
        # logger.debug("Sending LLM request", model=request.model, num_messages=len(messages))
        return await self.llm_aggregator.chat_completion(request)

    async def _use_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolOutput:
        """
        Helper method to invoke a tool via the ToolsRegistry.
        Includes a basic permission check placeholder.
        """
        if not self.tools_registry:
            logger.error(f"Agent '{self.agent_config.agent_id}' attempted to use tool '{tool_name}' but ToolsRegistry is not available.")
            return ToolOutput(error="ToolsRegistry not available to agent.", status_code=503) # Service Unavailable

        # --- Placeholder for Permission Check ---
        # In a complete system, the agent's RoleDefinition would list allowed tools.
        # This check would verify if `tool_name` is in `self.role_definition.tools`.
        # For example:
        # if self.role_definition and tool_name not in self.role_definition.tools:
        #     logger.warn(f"Agent '{self.agent_config.agent_id}' (Role: {self.agent_config.role_name}) "
        #                 f"attempted to use unauthorized tool '{tool_name}'.")
        #     return ToolOutput(error=f"Tool '{tool_name}' is not authorized for role {self.agent_config.role_name}.", status_code=403) # Forbidden

        # For now, we assume the agent has permission if the tool exists.
        # A more robust permission system would involve fetching the agent's RoleDefinition
        # (e.g., from CrewManager or a shared context) and checking its `tools` list.
        # This implies AgentConfig might need to store its RoleDefinition or have a way to access it.

        logger.info(f"Agent '{self.agent_config.agent_id}' (Role: {self.agent_config.role_name}) attempting to use tool '{tool_name}'.", tool_input_keys=list(tool_input.keys()))

        try:
            result = await self.tools_registry.invoke_tool(tool_name, tool_input)
            # logger.debug(f"Tool '{tool_name}' invoked by agent '{self.agent_config.agent_id}'. Result status: {result.status_code}")
            return result
        except Exception as e:
            # This catch is a safeguard, as invoke_tool itself should handle errors and return ToolOutput
            logger.error(f"Unexpected error during agent's _use_tool call for tool '{tool_name}'", error=str(e), exc_info=True)
            return ToolOutput(error=f"Unexpected system error invoking tool '{tool_name}': {str(e)}", status_code=500)


# Placeholder for Task and TaskContext if not already defined or to avoid circular imports
# This is just for type hinting within this file if direct imports cause issues.
# Actual imports should be resolved by Python's import system.
if "Task" not in globals(): # planning_structures.Task
    class Task: pass
if "TaskContext" not in globals(): # agent_structures.TaskContext
    class TaskContext: pass
if "AgentConfig" not in globals(): # agent_structures.AgentConfig
    class AgentConfig: agent_id: str; role_name: str; llm_config: Optional[Dict] # Simplified
if "TaskResult" not in globals(): # agent_structures.TaskResult
    class TaskResult: pass
if "LLMAggregator" not in globals(): # core.aggregator.LLMAggregator
    class LLMAggregator: async def chat_completion(self, req): pass # Simplified
if "ChatMessage" not in globals(): # models.ChatMessage
    class ChatMessage: pass
if "ChatCompletionResponse" not in globals(): # models.ChatCompletionResponse
    class ChatCompletionResponse: pass
if "ChatCompletionRequest" not in globals(): # models.ChatCompletionRequest
    class ChatCompletionRequest: pass
if "ToolsRegistry" not in globals(): # core.tools_registry.ToolsRegistry
    class ToolsRegistry: async def invoke_tool(self, name:str, inputs:dict): pass # Simplified
if "ToolOutput" not in globals(): # core.tool_interface.ToolOutput
    class ToolOutput: pass
# if "RoleDefinition" not in globals(): # agent_structures.RoleDefinition (for permission check)
#     class RoleDefinition: tools: List[str]
