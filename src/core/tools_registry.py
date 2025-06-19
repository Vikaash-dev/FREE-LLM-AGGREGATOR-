from typing import Dict, Any, Optional, Type as TypingType, List # Added Optional, TypingType, List
import structlog
from pydantic import BaseModel, ValidationError # Added BaseModel, ValidationError

from src.core.tool_interface import ToolInterface, ToolOutput # Import new interfaces

logger = structlog.get_logger(__name__)


class ToolsRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolInterface] = {} # Store instances of ToolInterface
        logger.info("ToolsRegistry initialized.")

    def register_tool(self, tool: ToolInterface):
        """Registers a tool instance."""
        if not isinstance(tool, ToolInterface):
            logger.error(f"Attempted to register invalid tool type: {type(tool)}. Tool must implement ToolInterface.")
            raise TypeError("Tool must implement ToolInterface.")

        if tool.name in self.tools:
            logger.warn(f"Tool '{tool.name}' is already registered. Overwriting.")

        self.tools[tool.name] = tool
        logger.info(f"Tool '{tool.name}' (type: {type(tool).__name__}) registered.")

    def unregister_tool(self, tool_name: str):
        """Unregisters a tool by its name."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Tool '{tool_name}' unregistered.")
        else:
            logger.warn(f"Attempted to unregister non-existent tool: {tool_name}")

    def get_tool(self, tool_name: str) -> Optional[ToolInterface]:
        """Retrieves a tool instance by its name."""
        tool = self.tools.get(tool_name)
        if not tool:
            logger.debug(f"Tool '{tool_name}' not found in registry.")
        return tool

    async def invoke_tool(self, tool_name: str, inputs_dict: Dict[str, Any]) -> ToolOutput:
        """
        Invokes a registered tool by its name with the provided input dictionary.
        Validates inputs against the tool's input_schema.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            logger.warn(f"Tool '{tool_name}' not found for invocation.")
            return ToolOutput(error=f"Tool '{tool_name}' not found.", status_code=404)

        try:
            # Validate inputs using the tool's specific input_schema Pydantic model
            validated_inputs: BaseModel = tool.input_schema(**inputs_dict)
            logger.debug(f"Inputs for tool '{tool_name}' validated successfully.", tool_inputs=inputs_dict)
        except ValidationError as e:
            logger.warn(f"Input validation error for tool '{tool_name}'", errors=e.errors(), inputs=inputs_dict)
            return ToolOutput(
                error=f"Input validation error for tool '{tool_name}': {e}",
                status_code=400
            )
        except Exception as e: # Catch other unexpected errors during input model instantiation
            logger.error(f"Unexpected error during input model instantiation for tool '{tool_name}'", error=str(e), exc_info=True)
            return ToolOutput(error=f"Unexpected error preparing inputs for tool '{tool_name}': {str(e)}", status_code=400)


        try:
            logger.info(f"Executing tool '{tool_name}' with validated inputs.")
            # Execute the tool's method with the validated Pydantic model instance
            result: ToolOutput = await tool.execute(validated_inputs)
            logger.info(f"Tool '{tool_name}' execution finished.", status_code=result.status_code, has_error=bool(result.error))
            return result
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}'", error=str(e), exc_info=True)
            return ToolOutput(
                error=f"Error executing tool '{tool_name}': {str(e)}",
                status_code=500 # Internal server error type
            )

    def get_tool_schema(self, tool_name: str) -> Optional[TypingType[BaseModel]]: # Changed return type
        """
        Returns the Pydantic input model (schema) for a given tool.
        """
        tool = self.get_tool(tool_name)
        if tool:
            return tool.input_schema
        logger.warn(f"Schema requested for unknown tool '{tool_name}'.")
        return None

    def get_tool_output_def(self, tool_name: str) -> Optional[TypingType[ToolOutput]]:
        """
        Returns the Pydantic output model definition (schema) for a given tool.
        """
        tool = self.get_tool(tool_name)
        if tool:
            return tool.output_schema
        logger.warn(f"Output schema requested for unknown tool '{tool_name}'.")
        return None


    def list_tools(self) -> List[Dict[str, str]]: # Changed return type for more info
        """Lists available tools with their names and descriptions."""
        return [{"name": name, "description": tool.description} for name, tool in self.tools.items()]

# Example of how a tool might be defined (conceptual, actual tools in separate files)
# class MyExampleTool(ToolInterface):
#     @property
#     def name(self) -> str: return "my_example_tool"
#     @property
#     def description(self) -> str: return "Does something amazing."
#     @property
#     def input_schema(self) -> TypingType[BaseModel]:
#         class MyToolInput(BaseModel):
#             param1: str
#             param2: int = 0
#         return MyToolInput
#     @property
#     def output_schema(self) -> TypingType[ToolOutput]: return ToolOutput
#
#     async def execute(self, inputs: BaseModel) -> ToolOutput:
#         # Assume inputs is MyToolInput after validation by ToolsRegistry
#         # cast_inputs = TypingType.cast(MyToolInput, inputs) # if needed, but registry should ensure this
#         return ToolOutput(output=f"Tool executed with {inputs.param1} and {inputs.param2}")

# if __name__ == '__main__':
#     # This is conceptual and would need async setup to run properly
#     # import asyncio
#     # registry = ToolsRegistry()
#     # example_tool_instance = MyExampleTool()
#     # registry.register_tool(example_tool_instance)
#     # print(registry.list_tools())
#     # schema = registry.get_tool_schema("my_example_tool")
#     # if schema:
#     #     print(schema.model_json_schema())
#     #
#     # async def run_invoke():
#     #     # Valid inputs
#     #     result_ok = await registry.invoke_tool("my_example_tool", {"param1": "hello"})
#     #     print("OK Result:", result_ok.model_dump_json(indent=2))
#     #     # Invalid inputs
#     #     result_bad = await registry.invoke_tool("my_example_tool", {"param1": 123}) # param1 should be str
#     #     print("Bad Input Result:", result_bad.model_dump_json(indent=2))
#     #     # Non-existent tool
#     #     result_notfound = await registry.invoke_tool("non_existent_tool", {})
#     #     print("Not Found Result:", result_notfound.model_dump_json(indent=2))
#     #
#     # asyncio.run(run_invoke())
