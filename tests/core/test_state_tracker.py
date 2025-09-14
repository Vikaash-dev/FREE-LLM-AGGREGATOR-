import unittest
from unittest.mock import patch, MagicMock
import datetime

from src.core.state_tracker import StateTracker

class TestStateTracker(unittest.TestCase):

    def setUp(self):
        # Mock structlog logger used by StateTracker to check calls
        self.mock_logger = MagicMock()
        # We need to patch where structlog.get_logger is called from within state_tracker.py
        self.logger_patch = patch('src.core.state_tracker.logger', self.mock_logger)
        self.logger_patch.start()
        # Instantiate the tracker *after* the patch is started
        self.tracker = StateTracker()

    def tearDown(self):
        self.logger_patch.stop()

    def test_state_tracker_initialization(self):
        self.assertIsNotNone(self.tracker)
        self.assertEqual(len(self.tracker.history), 0)
        self.mock_logger.info.assert_called_with("StateTracker initialized (Phase 1: Basic Logging).")

    def test_start_plan(self):
        self.tracker.start_plan("plan1", "Test instruction", 3)
        self.assertEqual(len(self.tracker.history), 1)
        event = self.tracker.history[0]
        self.assertEqual(event["event_type"], "plan_started")
        self.assertEqual(event["plan_id"], "plan1")
        self.assertEqual(event["user_instruction"], "Test instruction")
        self.mock_logger.info.assert_called_with("plan_started", plan_id="plan1", task_id=None,
                                                 user_instruction="Test instruction", total_tasks_in_plan=3)

    def test_start_task(self):
        self.tracker.start_task("plan1", "task1", "Do something", ["depA"])
        self.assertEqual(len(self.tracker.history), 1)
        event = self.tracker.history[0]
        self.assertEqual(event["event_type"], "task_started")
        self.assertEqual(event["task_id"], "task1")
        self.assertEqual(event["description"], "Do something")
        self.mock_logger.info.assert_called_with("task_started", plan_id="plan1", task_id="task1",
                                                 description="Do something", dependencies=["depA"])

    def test_update_task_reasoning_clarification_needed(self):
        decision = {"action": "NEEDS_CLARIFICATION", "details": "Needs more info", "confidence": 0.3}
        self.tracker.update_task_reasoning("plan1", "task1", decision)

        # Two events logged: task_reasoning_updated and task_clarification_needed
        self.assertEqual(len(self.tracker.history), 2)

        reasoning_event = self.tracker.history[0]
        self.assertEqual(reasoning_event["event_type"], "task_reasoning_updated")
        self.assertEqual(reasoning_event["reasoning_decision"], decision)
        self.mock_logger.info.assert_any_call("task_reasoning_updated", plan_id="plan1", task_id="task1",
                                                reasoning_decision=decision)

        clarification_event = self.tracker.history[1]
        self.assertEqual(clarification_event["event_type"], "task_clarification_needed")
        self.mock_logger.warn.assert_called_with("task_clarification_needed", plan_id="plan1", task_id="task1",
                                                   reason="Needs more info", confidence=0.3)


    def test_complete_task(self):
        self.tracker.complete_task("plan1", "task1", "COMPLETED", "Finished fine.", "Output generated.")
        self.assertEqual(len(self.tracker.history), 1)
        event = self.tracker.history[0]
        self.assertEqual(event["event_type"], "task_completed")
        self.assertEqual(event["final_status"], "COMPLETED")
        self.mock_logger.info.assert_called_with("task_completed", plan_id="plan1", task_id="task1",
                                                 final_status="COMPLETED", message="Finished fine.", output_summary="Output generated.")

    def test_fail_task(self):
        self.tracker.fail_task("plan1", "task1", "It broke", "SimError")
        self.assertEqual(len(self.tracker.history), 1)
        event = self.tracker.history[0]
        self.assertEqual(event["event_type"], "task_failed")
        self.assertEqual(event["error_message"], "It broke")
        self.mock_logger.warn.assert_called_with("task_failed", plan_id="plan1", task_id="task1",
                                                  error_message="It broke", error_type="SimError")

    def test_clear_history(self):
        self.tracker.start_task("p1", "t1", "desc")
        self.assertGreater(len(self.tracker.history), 0)
        self.tracker.clear_session_history()
        self.assertEqual(len(self.tracker.history), 0)
        self.mock_logger.info.assert_called_with("StateTracker in-memory history cleared.")


if __name__ == '__main__':
    unittest.main()
