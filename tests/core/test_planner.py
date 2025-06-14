import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock # For mocking LLMAggregator

from src.core.planner import DevikaInspiredPlanner
from src.core.planning_structures import ProjectContext, ExecutionPlan, TaskStatus
from src.models import ChatCompletionResponse, Choice, Message as OpenHandsMessage # Renamed to avoid conflict with unittest.Message

class TestDevikaInspiredPlanner(unittest.TestCase):

    def setUp(self):
        self.mock_llm_aggregator = AsyncMock()
        self.planner = DevikaInspiredPlanner(llm_aggregator=self.mock_llm_aggregator)
        self.project_context = ProjectContext(project_name="Test Project")

    def test_planner_initialization(self):
        self.assertIsNotNone(self.planner)
        self.assertEqual(self.planner.llm_aggregator, self.mock_llm_aggregator)

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_parse_user_intent_success(self):
        instruction = "Create a demo for a new feature."
        mock_response_content = '''
        {
            "goal": "Create a demonstration for a new product feature.",
            "entities": ["demonstration", "new product feature"],
            "constraints": [],
            "raw_instruction": "Create a demo for a new feature."
        }
        '''
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_response_content), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="sim_resp_1", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        result = self._run_async(self.planner.parse_user_intent(instruction, self.project_context))

        self.mock_llm_aggregator.chat_completion.assert_called_once()
        self.assertEqual(result["goal"], "Create a demonstration for a new product feature.")
        self.assertEqual(result["raw_instruction"], instruction)

    def test_parse_user_intent_llm_failure(self):
        instruction = "Another instruction."
        self.mock_llm_aggregator.chat_completion = AsyncMock(side_effect=Exception("LLM Sim Error"))

        result = self._run_async(self.planner.parse_user_intent(instruction, self.project_context))

        self.assertEqual(result["goal"], instruction) # Fallback
        self.assertIn("An error occurred: LLM Sim Error", result["constraints"][0])

    def test_decompose_complex_task_success(self):
        parsed_intent = {
            "goal": "Develop a new login page.",
            "entities": ["login page"],
            "constraints": ["Use React"],
            "raw_instruction": "Develop a new login page using React."
        }
        mock_task_list_json = '''
        [
            {"task_id": "task_1", "description": "Setup React environment", "dependencies": []},
            {"task_id": "task_2", "description": "Create login form component", "dependencies": ["task_1"]}
        ]
        '''
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_task_list_json), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="sim_resp_2", object="chat.completion", created=0, model="sim_model", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        execution_plan = self._run_async(self.planner.decompose_complex_task(parsed_intent, self.project_context))

        self.mock_llm_aggregator.chat_completion.assert_called_once()
        self.assertIsInstance(execution_plan, ExecutionPlan)
        self.assertEqual(len(execution_plan.tasks), 2)
        self.assertEqual(execution_plan.tasks[0].description, "Setup React environment")
        self.assertEqual(execution_plan.tasks[1].dependencies, ["task_1"])

    def test_decompose_complex_task_llm_json_error(self):
        parsed_intent = {"goal": "Goal for bad JSON", "raw_instruction": "Goal for bad JSON"}
        self.mock_llm_aggregator.chat_completion = AsyncMock(
            return_value=ChatCompletionResponse(id="sim_resp_3", object="chat.completion", created=0, model="sim_model",
                                             choices=[Choice(index=0, message=OpenHandsMessage(role="assistant", content="This is not JSON"), finish_reason="stop")])
        )
        execution_plan = self._run_async(self.planner.decompose_complex_task(parsed_intent, self.project_context))
        self.assertEqual(len(execution_plan.tasks), 1) # Fallback to single task
        self.assertEqual(execution_plan.tasks[0].description, "Goal for bad JSON")


if __name__ == '__main__':
    unittest.main()
