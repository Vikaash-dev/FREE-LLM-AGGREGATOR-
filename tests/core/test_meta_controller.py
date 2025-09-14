import unittest
import sqlite3
from unittest.mock import patch, MagicMock, AsyncMock

from src.core.meta_controller import MetaModelController, TaskComplexityAnalyzer, ExternalMemorySystem, ModelCapabilityProfile, TaskComplexity
from src.core.state_tracker import StateTracker
from src.models import ChatCompletionRequest, ChatMessage

class TestTaskComplexityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = TaskComplexityAnalyzer()

    def test_code_complexity(self):
        request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="write a python function to sort a list")])
        complexity = self.analyzer.analyze_task_complexity(request)
        self.assertGreaterEqual(complexity.domain_specificity, 0.3)

    def test_reasoning_complexity(self):
        request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="please think step by step and solve this logic puzzle")])
        complexity = self.analyzer.analyze_task_complexity(request)
        self.assertGreaterEqual(complexity.reasoning_depth, 0.5)

class TestExternalMemorySystem(unittest.TestCase):
    def setUp(self):
        # Use an in-memory database for testing
        self.memory_system = ExternalMemorySystem(db_path=":memory:")

    def tearDown(self):
        self.memory_system.close()

    def test_db_initialization(self):
        # Check if tables were created
        cursor = self.memory_system.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_patterns';")
        self.assertIsNotNone(cursor.fetchone())
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_performance';")
        self.assertIsNotNone(cursor.fetchone())

    def test_store_and_find_task_pattern(self):
        task_hash = "test_hash_123"
        task_type = "code"
        self.memory_system.store_task_pattern(
            task_hash=task_hash,
            task_type=task_type,
            complexity_features={"depth": 0.8},
            optimal_model="gpt-4",
            confidence_score=0.9
        )

        similar_tasks = self.memory_system.find_similar_tasks(task_hash="some_other_hash", task_type=task_type)
        self.assertEqual(len(similar_tasks), 1)
        self.assertEqual(similar_tasks[0]["optimal_model"], "gpt-4")

class TestMetaModelControllerInitialization(unittest.TestCase):
    def setUp(self):
        self.mock_state_tracker = MagicMock(spec=StateTracker)

    @patch('src.core.meta_controller.TORCH_AVAILABLE', False)
    def test_initialization_no_torch(self):
        """Test controller initialization when PyTorch is not available."""
        controller = MetaModelController(
            model_profiles={},
            state_tracker=self.mock_state_tracker,
            enable_ml_features=False
        )
        self.assertFalse(controller.ml_enabled)
        self.assertIsNone(controller.memory_system)

    @patch('src.core.meta_controller.TORCH_AVAILABLE', True)
    @patch('src.core.meta_controller.ExternalMemorySystem')
    def test_initialization_with_torch(self, mock_memory_system):
        """Test controller initialization when PyTorch is available."""
        controller = MetaModelController(
            model_profiles={},
            state_tracker=self.mock_state_tracker,
            enable_ml_features=True
        )
        self.assertTrue(controller.ml_enabled)
        mock_memory_system.assert_called_once()


class TestMetaModelControllerSelection(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_state_tracker = MagicMock(spec=StateTracker)

        self.model_profiles = {
            "cheap-model": ModelCapabilityProfile(
                model_name="cheap-model", provider="provider-a", size_category="small",
                reasoning_ability=0.5, code_generation=0.5, mathematical_reasoning=0.5,
                creative_writing=0.5, factual_knowledge=0.5, instruction_following=0.7,
                context_handling=0.6, avg_response_time=1.0, reliability_score=0.95,
                cost_per_token=0.0001, max_context_length=8192, domain_expertise=["general"],
                preferred_task_types=["general"],
            ),
            "powerful-model": ModelCapabilityProfile(
                model_name="powerful-model", provider="provider-b", size_category="large",
                reasoning_ability=0.9, code_generation=0.9, mathematical_reasoning=0.9,
                creative_writing=0.8, factual_knowledge=0.9, instruction_following=0.9,
                context_handling=0.9, avg_response_time=3.0, reliability_score=0.98,
                cost_per_token=0.0010, max_context_length=128000, domain_expertise=["code", "reasoning"],
                preferred_task_types=["complex_reasoning", "code_generation"],
            )
        }

        with patch('src.core.meta_controller.ExternalMemorySystem'), \
             patch('src.core.meta_controller.TaskComplexityAnalyzer'):
            self.controller = MetaModelController(
                model_profiles=self.model_profiles,
                state_tracker=self.mock_state_tracker,
                enable_ml_features=True
            )

        self.controller.complexity_analyzer = MagicMock(spec=TaskComplexityAnalyzer)
        self.controller.memory_system = MagicMock(spec=ExternalMemorySystem)
        self.controller.memory_system.find_similar_tasks = MagicMock(return_value=[])
        self.controller.memory_system.get_model_performance_history = MagicMock(return_value=[])

    async def test_select_simple_task_chooses_cheap_model(self):
        # Arrange
        simple_task = TaskComplexity(
            reasoning_depth=0.2, domain_specificity=0.1, context_length=100,
            computational_intensity=0.1, creativity_required=0.1, factual_accuracy_importance=0.2
        )
        self.controller.complexity_analyzer.analyze_task_complexity.return_value = simple_task
        request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hello")])

        # Act
        model_name, confidence = await self.controller.select_optimal_model(request)

        # Assert
        self.assertEqual(model_name, "cheap-model")
        self.assertGreater(confidence, 0)

    async def test_select_complex_task_chooses_powerful_model(self):
        # Arrange
        complex_task = TaskComplexity(
            reasoning_depth=0.9, domain_specificity=0.8, context_length=4000,
            computational_intensity=0.8, creativity_required=0.5, factual_accuracy_importance=0.9
        )
        self.controller.complexity_analyzer.analyze_task_complexity.return_value = complex_task
        request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="solve this complex math problem...")])

        # Act
        model_name, confidence = await self.controller.select_optimal_model(request)

        # Assert
        self.assertEqual(model_name, "powerful-model")
        self.assertGreater(confidence, 0)


if __name__ == '__main__':
    unittest.main()
