import unittest
import uuid
from src.core.planning_structures import (
    TaskStatus,
    ProjectContext,
    Task,
    ExecutionPlan,
    TaskResult
)

class TestPlanningStructures(unittest.TestCase):

    def test_task_status_enum(self):
        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertIn(TaskStatus.COMPLETED, list(TaskStatus))

    def test_project_context_creation(self):
        pc = ProjectContext(project_name="Test Project", project_description="A test desc.")
        self.assertEqual(pc.project_name, "Test Project")
        self.assertTrue(isinstance(pc.project_id, str))
        self.assertIsNotNone(uuid.UUID(pc.project_id)) # Check if it's a valid UUID string
        self.assertEqual(pc.additional_details, {})

    def test_task_creation(self):
        task_desc = "Test task description"
        t = Task(description=task_desc, raw_instruction="Do this")
        self.assertEqual(t.description, task_desc)
        self.assertEqual(t.raw_instruction, "Do this")
        self.assertEqual(t.status, TaskStatus.PENDING)
        self.assertTrue(isinstance(t.task_id, str))
        self.assertIsNotNone(uuid.UUID(t.task_id))
        self.assertEqual(t.dependencies, [])
        self.assertEqual(t.sub_tasks, [])

    def test_task_with_dependencies_and_subtasks(self):
        sub_task = Task(description="Sub-task")
        t = Task(description="Main task", dependencies=["dep1"], sub_tasks=[sub_task])
        self.assertEqual(t.dependencies, ["dep1"])
        self.assertEqual(len(t.sub_tasks), 1)
        self.assertEqual(t.sub_tasks[0].description, "Sub-task")

    def test_execution_plan_creation(self):
        user_instr = "Plan this"
        task1 = Task(description="Task 1")
        plan = ExecutionPlan(user_instruction=user_instr, tasks=[task1], parsed_intent={"goal": "Plan this"})
        self.assertEqual(plan.user_instruction, user_instr)
        self.assertEqual(plan.parsed_intent, {"goal": "Plan this"})
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].description, "Task 1")
        self.assertEqual(plan.overall_status, TaskStatus.PENDING)
        self.assertTrue(isinstance(plan.plan_id, str))
        self.assertIsNotNone(uuid.UUID(plan.plan_id))


    def test_task_result_creation(self):
        task_id = str(uuid.uuid4())
        tr = TaskResult(task_id=task_id, status=TaskStatus.COMPLETED, message="Done", output={"data": 123})
        self.assertEqual(tr.task_id, task_id)
        self.assertEqual(tr.status, TaskStatus.COMPLETED)
        self.assertEqual(tr.message, "Done")
        self.assertEqual(tr.output, {"data": 123})

if __name__ == '__main__':
    unittest.main()
