from typing import Dict, List, Any, Optional, Callable
import structlog

from src.core.workflow_structures import WorkflowGraph, WorkflowNode, WorkflowEdge, NodeType
from src.core.crew_manager import CrewManager # For CREW_TASK nodes
from src.core.aggregator import LLMAggregator # For router nodes or system tasks
# from src.core.base_agent import AbstractBaseAgent # For AGENT_TASK nodes (needs agent lookup)
# from src.core.agent_structures import TaskContext # For AGENT_TASK / CREW_TASK
# from src.core.planning_structures import Task, ExecutionPlan # For AGENT_TASK / CREW_TASK

logger = structlog.get_logger(__name__)

class WorkflowManager:
    def __init__(self, crew_manager: CrewManager, llm_aggregator: LLMAggregator):
        self.crew_manager = crew_manager
        self.llm_aggregator = llm_aggregator # May be used by certain node types
        self.workflows: Dict[str, WorkflowGraph] = {}
        # Conditional routers map a name (specified in a CONDITIONAL_ROUTER node's config)
        # to a Python function that takes the current global_state and returns an edge_label.
        self.conditional_routers: Dict[str, Callable[[Dict[str, Any]], str]] = {}
        logger.info("WorkflowManager initialized.")

    def load_workflow(self, workflow_graph: WorkflowGraph) -> bool:
        """
        Loads and validates a workflow graph.
        """
        if not workflow_graph.graph_id:
            logger.error("Workflow graph has no ID. Cannot load.")
            return False

        if not workflow_graph.start_node_id:
            logger.error(f"Workflow '{workflow_graph.graph_id}' has no start_node_id defined.")
            return False

        if workflow_graph.start_node_id not in workflow_graph.nodes:
            logger.error(f"Start node '{workflow_graph.start_node_id}' for workflow '{workflow_graph.graph_id}' not found in nodes collection.")
            return False

        # Basic validation: ensure all edge node IDs exist
        for edge in workflow_graph.edges:
            if edge.source_node_id not in workflow_graph.nodes:
                logger.error(f"Edge source node '{edge.source_node_id}' not found in nodes for workflow '{workflow_graph.graph_id}'.")
                return False
            if edge.target_node_id not in workflow_graph.nodes:
                logger.error(f"Edge target node '{edge.target_node_id}' not found in nodes for workflow '{workflow_graph.graph_id}'.")
                return False

        self.workflows[workflow_graph.graph_id] = workflow_graph
        logger.info(f"Workflow '{workflow_graph.graph_id}' loaded and validated successfully.")
        return True

    def register_conditional_router(self, router_name: str, router_function: Callable[[Dict[str, Any]], str]):
        """
        Registers a Python function to be used as a conditional router.
        The function should take the global_state dict and return a string (edge_label).
        """
        if not callable(router_function):
            logger.error(f"Attempted to register non-callable router function for '{router_name}'.")
            raise TypeError("Router function must be callable.")

        self.conditional_routers[router_name] = router_function
        logger.info(f"Conditional router '{router_name}' registered with function: {router_function.__name__}.")

    async def process_workflow(self, workflow_id: str, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Processing workflow '{workflow_id}' with initial input keys: {list(initial_input.keys())}")

        graph = self.workflows.get(workflow_id)
        if not graph:
            logger.error(f"Workflow '{workflow_id}' not found.")
            return {"error": f"Workflow '{workflow_id}' not found", "final_status": "ERROR"}

        if not graph.start_node_id: # Should be caught by load_workflow, but defensive check
            logger.error(f"Workflow '{workflow_id}' has no start node defined.")
            return {"error": "Workflow has no start node", "final_status": "ERROR"}

        global_state: Dict[str, Any] = initial_input.copy()
        current_node_id: Optional[str] = graph.start_node_id
        visited_nodes_in_path = set() # For simple loop detection in current path segment
        max_steps = 20  # Safety break for non-branching paths or unhandled loops
        step_count = 0

        while current_node_id and step_count < max_steps:
            node = graph.nodes.get(current_node_id)
            if not node:
                logger.error(f"Node ID '{current_node_id}' not found in workflow '{workflow_id}'. Halting.")
                global_state["error"] = f"Node '{current_node_id}' definition not found."
                global_state["final_status"] = "ERROR"
                break

            logger.info(f"Step {step_count + 1}: Executing Node ID '{node.id}', Type '{node.node_type.value}', Name '{node.name}'.")

            # Simple loop detection for non-router nodes
            if node.node_type != NodeType.CONDITIONAL_ROUTER and current_node_id in visited_nodes_in_path:
                logger.warn(f"Re-visiting node '{current_node_id}' which is not a router. Potential loop detected. Halting workflow.")
                global_state["error"] = f"Loop detected: re-visiting node '{current_node_id}'."
                global_state["final_status"] = "ERROR_LOOP_DETECTED"
                break
            visited_nodes_in_path.add(current_node_id) # Add to current path

            # Prepare node input from global_state based on node.input_keys
            node_input = {k: global_state.get(k) for k in node.input_keys if k in global_state}
            logger.debug(f"Node '{node.id}' input: {node_input}")

            node_output_data: Dict[str, Any] = {} # Data produced by this node

            if node.node_type == NodeType.START:
                logger.debug(f"Start node '{node.id}': Passing initial input to output keys.")
                # Start node typically passes initial_input keys matching its output_keys
                for key in node.output_keys:
                    if key in global_state: # If initial_input provided this key
                        node_output_data[key] = global_state[key]
                    else: # Or if config has defaults for output_keys for a start node
                        node_output_data[key] = node.config.get(key)
                next_node_id_override = None


            elif node.node_type == NodeType.END:
                logger.info(f"Reached END node '{node.id}'. Workflow processing finished.")
                global_state["final_status"] = "COMPLETED"
                # Populate global_state with final node inputs if specified (as a form of return value mapping)
                for key in node.input_keys:
                    if key in global_state:
                        global_state[f"final_output_{key}"] = global_state[key] # Example: prefix final outputs
                current_node_id = None # End of workflow
                break # Exit while loop

            elif node.node_type == NodeType.AGENT_TASK:
                # Placeholder: Integration with a single agent execution.
                # Requires mapping node.config (e.g., role_name, task_description_template)
                # and node_input to an agent's execute_task.
                logger.warn(f"AGENT_TASK node '{node.id}' execution not fully implemented. Using mock output.")
                # Simulate output based on output_keys
                for i, key in enumerate(node.output_keys):
                    node_output_data[key] = f"mock_agent_output_for_{key}_from_node_{node.id}"
                # Simulate success for now
                if not node.output_keys: # If no output keys, maybe it signals success via a state var
                    global_state[f"{node.id}_status"] = "COMPLETED"


            elif node.node_type == NodeType.CREW_TASK:
                # Placeholder: Integration with CrewManager.process_execution_plan.
                # Requires mapping node.config (e.g., plan_definition or reference)
                # and node_input to the plan's initial context.
                logger.warn(f"CREW_TASK node '{node.id}' execution not fully implemented. Using mock output.")
                for i, key in enumerate(node.output_keys):
                     node_output_data[key] = f"mock_crew_output_for_{key}_from_node_{node.id}"
                if not node.output_keys:
                    global_state[f"{node.id}_status"] = "COMPLETED"


            elif node.node_type == NodeType.CONDITIONAL_ROUTER:
                visited_nodes_in_path.clear() # Reset for new path segment after router
                router_func_name = node.config.get("router_function_name")
                if router_func_name and router_func_name in self.conditional_routers:
                    try:
                        router_function = self.conditional_routers[router_func_name]
                        next_edge_label = router_function(global_state) # Pass current global_state
                        logger.info(f"Router '{router_func_name}' for node '{node.id}' decided on edge_label: '{next_edge_label}'.")

                        found_next_edge = False
                        for edge in graph.get_outgoing_edges(current_node_id):
                            if edge.edge_label == next_edge_label:
                                current_node_id = edge.target_node_id
                                found_next_edge = True
                                logger.debug(f"Router directs to next node: '{current_node_id}' via edge label '{next_edge_label}'.")
                                break
                        if not found_next_edge:
                            err_msg = f"Router condition '{next_edge_label}' for node '{node.id}' did not match any outgoing edge labels."
                            logger.error(err_msg)
                            global_state["error"] = err_msg
                            global_state["final_status"] = "ERROR_ROUTING"
                            current_node_id = None
                        # continue to next iteration without standard next node logic
                        step_count += 1
                        continue
                    except Exception as e:
                        err_msg = f"Error executing router function '{router_func_name}' for node '{node.id}': {str(e)}"
                        logger.error(err_msg, exc_info=True)
                        global_state["error"] = err_msg
                        global_state["final_status"] = "ERROR_ROUTING_EXECUTION"
                        current_node_id = None
                else:
                    err_msg = f"Conditional router function '{router_func_name}' not found or not specified in config for node '{node.id}'."
                    logger.error(err_msg)
                    global_state["error"] = err_msg
                    global_state["final_status"] = "ERROR_CONFIG"
                    current_node_id = None
            else:
                logger.warn(f"Node type '{node.node_type.value}' for node '{node.id}' not fully supported yet. Skipping actual execution logic.")
                # Mock output for progression
                for key in node.output_keys:
                    node_output_data[key] = f"mock_output_for_{key}_unsupported_type"

            # Update global_state with node's output
            for key, value in node_output_data.items():
                if key in node.output_keys: # Only update if it's an expected output key
                    global_state[key] = value
            logger.debug(f"Node '{node.id}' output: {node_output_data}, updated global_state keys: {list(node_output_data.keys())}")

            # Determine next node (for non-router nodes or routers that didn't 'continue')
            if current_node_id: # Check if router logic already nulled it on error
                next_node_id_temp = None
                # Simple case: find the first non-conditional outgoing edge
                # More complex scenarios might involve multiple outgoing edges based on conditions not handled by CONDITIONAL_ROUTER type.
                for edge in graph.get_outgoing_edges(current_node_id):
                    if not edge.condition_name and not edge.edge_label: # Simple direct edge
                        next_node_id_temp = edge.target_node_id
                        logger.debug(f"Found direct next node: '{next_node_id_temp}' from '{current_node_id}'.")
                        break
                current_node_id = next_node_id_temp
                if not current_node_id:
                    logger.info(f"No further outgoing unconditional edge from node '{node.id}'. Workflow may end here if not an END node.")

            step_count += 1

        if step_count >= max_steps:
            logger.warn(f"Workflow '{workflow_id}' processing reached max steps ({max_steps}). Halting.")
            global_state["error"] = "Max steps reached during workflow execution."
            if "final_status" not in global_state: global_state["final_status"] = "ERROR_MAX_STEPS"

        if "final_status" not in global_state: # If loop exited due to no current_node_id but not via END node
            logger.warn(f"Workflow '{workflow_id}' ended without reaching an END node. Current node was '{node.id if node else 'None'}'.")
            global_state["final_status"] = "INCOMPLETE"


        logger.info(f"Workflow '{workflow_id}' processing finished. Final status: {global_state.get('final_status', 'UNKNOWN')}. Final state keys: {list(global_state.keys())}")
        return global_state
