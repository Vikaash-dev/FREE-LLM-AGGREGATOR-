import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.core.reasoner import ContextualReasoningEngine
from src.core.planning_structures import Task, ProjectContext, TaskStatus
from src.models import ChatCompletionResponse, ChatCompletionChoice as Choice, ChatMessage as OpenHandsMessage


class TestContextualReasoner(unittest.TestCase):

    def setUp(self):
        self.mock_llm_aggregator = AsyncMock()
        self.reasoner = ContextualReasoningEngine(llm_aggregator=self.mock_llm_aggregator)
        self.project_context = ProjectContext(project_name="Test Project", additional_details={"git_branch": "main"})
        self.sample_task = Task(description="Refactor database module", raw_instruction="Refactor DB")

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_reasoner_initialization(self):
        self.assertIsNotNone(self.reasoner)

    # Changed to test_analyze_context_run to use the helper, or ensure an async test runner is used
    def test_analyze_context_run(self):
        analyzed_context = self._run_async(self.reasoner.analyze_context(self.sample_task, self.project_context))
        self.assertEqual(analyzed_context["task_id"], self.sample_task.task_id)
        self.assertEqual(analyzed_context["task_description"], self.sample_task.description)
        self.assertEqual(analyzed_context["project_name"], "Test Project")
        self.assertEqual(analyzed_context["project_additional_details"], {"git_branch": "main"})

    def test_reason_about_task_success(self):
        analyzed_context_sample = {"task_id": self.sample_task.task_id, "project_name": "Test Project"}
        mock_reasoning_json = '''
        {
            "reasoning_steps": ["Step 1", "Step 2"],
            "potential_issues": ["Issue A"],
            "information_needed": ["Info X"],
            "suggested_approach": "Approach Y",
            "confidence_score": 0.85,
            "requires_clarification": false
        }
        '''
        mock_llm_choice = Choice(index=0, message=OpenHandsMessage(role="assistant", content=mock_reasoning_json), finish_reason="stop")
        mock_llm_response = ChatCompletionResponse(id="sim_resp_r1", object="chat.completion", created=0, model="sim_model", provider="mock", choices=[mock_llm_choice])
        self.mock_llm_aggregator.chat_completion = AsyncMock(return_value=mock_llm_response)

        reasoning_output = self._run_async(self.reasoner.reason_about_task(self.sample_task, analyzed_context_sample))

        self.mock_llm_aggregator.chat_completion.assert_called_once()
        self.assertEqual(reasoning_output["confidence_score"], 0.85)
        self.assertFalse(reasoning_output["requires_clarification"])

    def test_make_decision_proceed(self):
        reasoning_output = {
            "confidence_score": 0.9,
            "requires_clarification": False,
            "suggested_approach": "Do it."
        }
        decision = self.reasoner.make_decision(reasoning_output)
        self.assertEqual(decision["action"], "PROCEED")
        self.assertTrue("Do it." in decision["details"])

    def test_make_decision_needs_clarification_low_confidence(self):
        reasoning_output = {"confidence_score": 0.4, "requires_clarification": False}
        decision = self.reasoner.make_decision(reasoning_output)
        self.assertEqual(decision["action"], "NEEDS_CLARIFICATION")

    def test_make_decision_needs_clarification_flagged(self):
        reasoning_output = {"confidence_score": 0.8, "requires_clarification": True}
        decision = self.reasoner.make_decision(reasoning_output)
        self.assertEqual(decision["action"], "NEEDS_CLARIFICATION")

if __name__ == '__main__':
    unittest.main()
