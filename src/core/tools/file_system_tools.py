from pathlib import Path
from typing import Type as TypingType, Any, Dict # Use TypingType
import structlog

from pydantic import BaseModel, Field

from src.core.tool_interface import ToolInterface, ToolOutput

logger = structlog.get_logger(__name__)

# --- ReadFileTool ---
class ReadFileToolInput(BaseModel):
    file_path: str = Field(description="The path to the file that needs to be read.")

class ReadFileTool(ToolInterface):
    # Using _val suffix to avoid potential conflicts if Pydantic models themselves
    # have attributes named 'name', 'description', etc. in some advanced scenarios.
    # For simple properties, direct assignment in __init__ or class level is also fine.
    name_val: str = "file_read"
    description_val: str = "Reads the entire content of a specified file and returns it as a string."
    input_schema_val: TypingType[BaseModel] = ReadFileToolInput
    # The output of successful execution will be a string, placed in ToolOutput.output
    output_schema_val: TypingType[ToolOutput] = ToolOutput

    @property
    def name(self) -> str:
        return self.name_val

    @property
    def description(self) -> str:
        return self.description_val

    @property
    def input_schema(self) -> TypingType[BaseModel]:
        return self.input_schema_val

    @property
    def output_schema(self) -> TypingType[ToolOutput]:
        return self.output_schema_val

    async def execute(self, inputs: ReadFileToolInput) -> ToolOutput:
        logger.info(f"Executing ReadFileTool for path: {inputs.file_path}")
        try:
            # Security consideration: In a real system, ensure file_path is not traversing
            # to unauthorized directories. This might involve resolving the path
            # and checking if it's within a permitted base directory.
            # For this example, we assume the path is safe.
            target_path = Path(inputs.file_path)
            if not target_path.is_file():
                logger.warn(f"File not found at path: {inputs.file_path}")
                return ToolOutput(error=f"File not found: {inputs.file_path}", status_code=404)

            content = target_path.read_text(encoding='utf-8')
            logger.info(f"Successfully read file: {inputs.file_path}, content length: {len(content)}")
            return ToolOutput(output=content, status_code=200)
        except FileNotFoundError: # Should be caught by is_file() check, but as a safeguard
            logger.warn(f"ReadFileTool FileNotFoundError: {inputs.file_path}")
            return ToolOutput(error=f"File not found: {inputs.file_path}", status_code=404)
        except Exception as e:
            logger.error(f"Error reading file '{inputs.file_path}'", error=str(e), exc_info=True)
            return ToolOutput(error=f"Error reading file '{inputs.file_path}': {str(e)}", status_code=500)


# --- WriteFileTool ---
class WriteFileToolInput(BaseModel):
    file_path: str = Field(description="The path to the file where content should be written. Existing file will be overwritten.")
    content: str = Field(description="The string content to write to the file.")

class WriteFileTool(ToolInterface):
    name_val: str = "file_write"
    description_val: str = "Writes the given string content to a specified file. Overwrites if the file exists."
    input_schema_val: TypingType[BaseModel] = WriteFileToolInput
    # Output for success is a simple confirmation message
    output_schema_val: TypingType[ToolOutput] = ToolOutput

    @property
    def name(self) -> str:
        return self.name_val

    @property
    def description(self) -> str:
        return self.description_val

    @property
    def input_schema(self) -> TypingType[BaseModel]:
        return self.input_schema_val

    @property
    def output_schema(self) -> TypingType[ToolOutput]:
        return self.output_schema_val

    async def execute(self, inputs: WriteFileToolInput) -> ToolOutput:
        logger.info(f"Executing WriteFileTool for path: {inputs.file_path}")
        try:
            # Security considerations for file path apply here as well.
            # Additionally, ensure the agent has permissions to write to this location.
            target_path = Path(inputs.file_path)

            # Ensure parent directory exists, create if not.
            # This is a common convenience for file writing tools.
            target_path.parent.mkdir(parents=True, exist_ok=True)

            target_path.write_text(inputs.content, encoding='utf-8')
            logger.info(f"Successfully wrote to file: {inputs.file_path}")
            return ToolOutput(output={'message': f"File '{inputs.file_path}' written successfully."}, status_code=200)
        except Exception as e:
            logger.error(f"Error writing file '{inputs.file_path}'", error=str(e), exc_info=True)
            return ToolOutput(error=f"Error writing file '{inputs.file_path}': {str(e)}", status_code=500)
