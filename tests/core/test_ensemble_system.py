import unittest
import asyncio
from unittest.mock import MagicMock, patch
from datetime import datetime, UTC

from src.core.ensemble_system import EnsembleSystem, ResponseCandidate
from src.models import ChatCompletionRequest, ChatCompletionResponse, ChatMessage

class TestEnsembleSystem(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ensemble_system = EnsembleSystem()
        self.request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Test")])
        now = int(datetime.now(UTC).timestamp())

        # Create mock responses with all required fields
        self.response1 = ChatCompletionResponse(
            id="resp1", created=now, model="model-1", provider="provider-a",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "Response 1"},
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        )
        self.response2 = ChatCompletionResponse(
            id="resp2", created=now, model="model-2", provider="provider-b",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "Response 2"},
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
        )
        self.response3 = ChatCompletionResponse(
            id="resp3", created=now, model="model-3", provider="provider-c",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "Response 3"},
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": 8, "completion_tokens": 7, "total_tokens": 15}
        )

        # Create mock metadata
        self.metadata1 = {"confidence": 0.9, "response_time": 0.5, "cost_estimate": 0.001}
        self.metadata2 = {"confidence": 0.8, "response_time": 0.8, "cost_estimate": 0.002}
        self.metadata3 = {"confidence": 0.95, "response_time": 0.4, "cost_estimate": 0.0005} # Best

    @patch('src.core.ensemble_system.ResponseQualityEvaluator.evaluate_response')
    async def test_generate_ensemble_response(self, mock_evaluate_response):
        # Arrange
        # Mock the quality evaluator to return different scores for each response
        def side_effect(response, request):
            if response.id == "resp1":
                return {'coherence_score': 0.8, 'relevance_score': 0.9, 'factual_accuracy_score': 0.85, 'creativity_score': 0.7, 'safety_score': 0.95}
            elif response.id == "resp2":
                return {'coherence_score': 0.7, 'relevance_score': 0.7, 'factual_accuracy_score': 0.75, 'creativity_score': 0.6, 'safety_score': 0.9}
            elif response.id == "resp3":
                return {'coherence_score': 0.85, 'relevance_score': 0.95, 'factual_accuracy_score': 0.9, 'creativity_score': 0.8, 'safety_score': 0.98}
            return {}
        mock_evaluate_response.side_effect = side_effect

        model_responses = {
            "model-1": self.response1,
            "model-2": self.response2,
            "model-3": self.response3
        }
        model_metadata = {
            "model-1": self.metadata1,
            "model-2": self.metadata2,
            "model-3": self.metadata3
        }

        # Act
        final_response = await self.ensemble_system.generate_ensemble_response(
            self.request, model_responses, model_metadata
        )

        # Assert
        self.assertIsNotNone(final_response)
        # It should pick the best response (response 3)
        self.assertEqual(final_response.id, "resp3")
        self.assertEqual(final_response.choices[0].message.content, "Response 3")


if __name__ == '__main__':
    unittest.main()
