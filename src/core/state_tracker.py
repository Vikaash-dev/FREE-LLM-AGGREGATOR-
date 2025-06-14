from typing import Dict, Any, List, Optional
import datetime
import structlog

# Assuming TaskResult and other relevant structures might be useful for context,
# but for Phase 1, methods will primarily log information.
# from .planning_structures import TaskResult (Import if TaskResult object is passed directly)

logger = structlog.get_logger(__name__)

class StateTracker:
    '''
    A simple state tracker for Phase 1 to log the lifecycle of tasks
    within an execution plan.

    For Phase 1, this tracker logs to console/standard output.
    It does not persist state or offer visualization.
    '''

    def __init__(self):
        self.history: List[Dict[str, Any]] = [] # Optional: for in-memory history during a session
        logger.info("StateTracker initialized (Phase 1: Basic Logging).")

    def _log_event(self, event_type: str, plan_id: Optional[str], task_id: Optional[str], details: Dict[str, Any]):
        timestamp = datetime.datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "plan_id": plan_id,
            "task_id": task_id,
            **details # Spread the details dictionary
        }
        self.history.append(log_entry) # Store in memory for potential inspection during a session

        # Log using structlog, ensuring all context from details is passed as key-value
        log_func = logger.info
        if event_type.endswith("_failed") or event_type == "task_clarification_needed":
            log_func = logger.warn

        log_func(event_type, plan_id=plan_id, task_id=task_id, **details)


    def start_plan(self, plan_id: str, user_instruction: str, num_tasks: int):
        details = {"user_instruction": user_instruction, "total_tasks_in_plan": num_tasks}
        self._log_event("plan_started", plan_id=plan_id, task_id=None, details=details)

    def complete_plan(self, plan_id: str, final_status: str, total_execution_time: Optional[float] = None):
        details = {"final_plan_status": final_status}
        if total_execution_time is not None:
            details["total_execution_time_seconds"] = round(total_execution_time, 2)
        self._log_event("plan_completed", plan_id=plan_id, task_id=None, details=details)

    def start_task(self, plan_id: str, task_id: str, task_description: str, dependencies: Optional[List[str]] = None):
        details = {"description": task_description, "dependencies": dependencies or []}
        self._log_event("task_started", plan_id=plan_id, task_id=task_id, details=details)

    def update_task_reasoning(self, plan_id: str, task_id: str, reasoning_decision: Dict[str, Any]):
        # decision_action = reasoning_decision.get('action')
        # decision_details = reasoning_decision.get('details')
        # decision_confidence = reasoning_decision.get('confidence')
        # log_details = {
        #     "decision_action": decision_action,
        #     "decision_details": decision_details, # Can be long
        #     "decision_confidence": decision_confidence
        # }
        # For Phase 1, let's pass the whole decision object for simplicity in logging details
        self._log_event("task_reasoning_updated", plan_id=plan_id, task_id=task_id, details={"reasoning_decision": reasoning_decision})

        if reasoning_decision.get('action') == "NEEDS_CLARIFICATION":
            clarification_details = {
                "reason": reasoning_decision.get('details', "No specific reason provided."),
                "confidence": reasoning_decision.get('confidence', 0.0)
            }
            self._log_event("task_clarification_needed", plan_id=plan_id, task_id=task_id, details=clarification_details)


    def complete_task(self, plan_id: str, task_id: str, status: str, message: Optional[str] = None, output_summary: Optional[str] = None):
        details = {"final_status": status, "message": message}
        if output_summary:
            details["output_summary"] = output_summary
        self._log_event("task_completed", plan_id=plan_id, task_id=task_id, details=details)

    def fail_task(self, plan_id: str, task_id: str, error: str, error_type: Optional[str] = None):
        details = {"error_message": error}
        if error_type:
            details["error_type"] = error_type
        self._log_event("task_failed", plan_id=plan_id, task_id=task_id, details=details)

    def get_session_history(self) -> List[Dict[str, Any]]:
        '''Returns the in-memory history of events for the current session.'''
        return self.history

    def clear_session_history(self):
        '''Clears the in-memory history.'''
        self.history = []
        logger.info("StateTracker in-memory history cleared.")
