# Graph-Based Workflow Management System

## Introduction

The Graph-Based Workflow Management System in OpenHands provides a flexible and powerful way to define, manage, and execute complex, multi-step processes that may involve various agents, crews of agents, conditional logic, and other custom operations. This system allows for the visual design (conceptually) and programmatic execution of workflows as directed acyclic graphs (DAGs) or cyclic graphs (with controlled looping via routers).

Workflows are defined by nodes representing specific operations and edges representing the flow of control and data between these operations. This system aims to provide a higher-level orchestration layer above individual agent tasks or crew executions.

## Core Concepts & Data Models

The workflow system is built upon the following Pydantic models defined in `src.core.workflow_structures`:

*   **`NodeType(str, Enum)`**: Defines the different types of operations a node can represent:
    *   `START ("start_node")`: The entry point of a workflow.
    *   `END ("end_node")`: A terminal point of a workflow.
    *   `AGENT_TASK ("agent_task")`: A task to be executed by a single, specific agent role.
    *   `CREW_TASK ("crew_task")`: A more complex task or sub-plan to be executed by a `CrewManager` (involving multiple agents).
    *   `CONDITIONAL_ROUTER ("conditional_router_node")`: A node that directs the workflow along different paths based on the current `global_state` and a registered router function.

*   **`WorkflowNode(BaseModel)`**: Represents an individual step or operation in the workflow.
    *   `id: str`: Unique identifier for the node within the graph.
    *   `node_type: NodeType`: The type of the node (from `NodeType` enum).
    *   `name: str`: A human-readable name for the node.
    *   `config: Dict[str, Any]`: Configuration specific to the node's type.
        *   For `AGENT_TASK`: Might include `required_role`, `task_description_template`.
        *   For `CREW_TASK`: Might include `crew_id_or_config_ref`, `plan_description_template`.
        *   For `CONDITIONAL_ROUTER`: Must include `router_function_name` which maps to a registered Python callable.
    *   `input_keys: List[str]`: Keys from the `global_state` that are required as input for this node.
    *   `output_keys: List[str]`: Keys that this node will produce as output, which will be merged back into the `global_state`.

*   **`WorkflowEdge(BaseModel)`**: Defines a directed connection between two nodes.
    *   `source_node_id: str`: The ID of the node where the edge originates.
    *   `target_node_id: str`: The ID of the node where the edge terminates.
    *   `condition_name: Optional[str]`: (Less used currently) Could be used if a node's output directly sets a boolean condition by this name.
    *   `edge_label: Optional[str]`: A label for the edge. Crucially used by `CONDITIONAL_ROUTER` nodes. The router function's string output must match this `edge_label` for the workflow to proceed along this edge.

*   **`WorkflowGraph(BaseModel)`**: Represents the entire workflow definition.
    *   `graph_id: str`: Unique identifier for the workflow.
    *   `description: Optional[str]`: A human-readable description of the workflow.
    *   `nodes: Dict[str, WorkflowNode]`: A dictionary of all nodes in the graph, keyed by their `id`.
    *   `edges: List[WorkflowEdge]`: A list of all edges connecting the nodes.
    *   `start_node_id: Optional[str]`: The ID of the node where workflow execution begins. This node must exist in the `nodes` dictionary.
    *   `global_state_schema: Optional[Dict[str, Any]]`: (Future use) A JSON schema to define the expected structure and types of the `global_state` dictionary that is passed between nodes.

## `WorkflowManager` API

The `WorkflowManager` (in `src.core.workflow_manager`) is responsible for loading, managing, and executing workflow graphs.

*   **`__init__(self, crew_manager: CrewManager, llm_aggregator: LLMAggregator)`**:
    *   Initializes the manager. It requires a `CrewManager` instance (for executing `CREW_TASK` nodes) and an `LLMAggregator` (which might be used by certain node types like advanced routers or system-level AI tasks, though not heavily used by the current placeholder nodes).

*   **`load_workflow(self, workflow_graph: WorkflowGraph) -> bool`**:
    *   Adds a `WorkflowGraph` definition to the manager's internal store.
    *   Performs basic validation (e.g., presence of `start_node_id`, existence of nodes referenced in edges).
    *   Returns `True` if loaded successfully, `False` otherwise.

*   **`register_conditional_router(self, router_name: str, router_function: Callable[[Dict[str, Any]], str])`**:
    *   Registers a Python callable function that can be used by `CONDITIONAL_ROUTER` nodes.
    *   `router_name`: A string name that will be referenced in a `CONDITIONAL_ROUTER` node's `config.router_function_name`.
    *   `router_function`: A Python function that accepts the current `global_state` (a dictionary) and returns a string. This string output is used as an `edge_label` to select the next path from the router node.

*   **`async def process_workflow(self, workflow_id: str, initial_input: Dict[str, Any]) -> Dict[str, Any]`**:
    *   This is the main method to execute a loaded workflow.
    *   `workflow_id`: The ID of the `WorkflowGraph` to execute.
    *   `initial_input`: A dictionary representing the initial state/data for the workflow. This becomes the starting `global_state`.
    *   **Execution Flow:**
        1.  Initializes `global_state` with `initial_input`.
        2.  Starts execution from the `graph.start_node_id`.
        3.  Iteratively processes nodes:
            a.  Retrieves the current node.
            b.  Performs basic loop detection.
            c.  Gathers inputs for the node from `global_state` based on `node.input_keys`.
            d.  Executes the node based on its `node_type`:
                *   `START`: Passes specified inputs to outputs.
                *   `END`: Terminates the workflow successfully.
                *   `AGENT_TASK` / `CREW_TASK`: (Currently placeholders) Would delegate to `CrewManager` or a specific agent. The results (e.g., from `TaskResult.output`) would be mapped to `node.output_keys` and merged into `global_state`.
                *   `CONDITIONAL_ROUTER`: Retrieves `router_function_name` from `node.config`. Calls the corresponding registered router function with `global_state`. The string returned by the router function is used to find an outgoing edge with a matching `edge_label`. If a match is found, `current_node_id` is updated to the edge's `target_node_id`, and execution continues. If no match or router error, the workflow halts.
            e.  Merges the node's output (if any, as defined by `node.output_keys`) back into `global_state`.
            f.  Determines the next node by finding an unconditional outgoing edge (for non-router nodes).
        4.  Includes a safety break (`max_steps`) to prevent infinite loops in simple paths.
        5.  Returns the final `global_state` dictionary, which will include any outputs produced by the workflow and a `final_status` field (e.g., "COMPLETED", "ERROR", "INCOMPLETE").

## Conceptual Example

### Workflow Definition (e.g., as Python dict or loaded from YAML/JSON)

```python
# Conceptual Python dictionary representation of a workflow graph
example_workflow_dict = {
    "graph_id": "simple_approval_workflow",
    "description": "A workflow that gets an approval status and then branches.",
    "start_node_id": "node_start",
    "nodes": {
        "node_start": {
            "id": "node_start", "node_type": "start_node", "name": "Start Workflow",
            "output_keys": ["user_request"] # Assumes 'user_request' is in initial_input
        },
        "node_get_approval": {
            "id": "node_get_approval", "node_type": "agent_task", "name": "Get Approval Status",
            "config": {"required_role": "ApprovalAgent", "task_description_template": "Review request: {{user_request}} for approval."},
            "input_keys": ["user_request"],
            "output_keys": ["approval_status", "approver_comments"] # e.g., approval_status = "approved" or "rejected"
        },
        "node_approval_router": {
            "id": "node_approval_router", "node_type": "conditional_router_node", "name": "Check Approval",
            "config": {"router_function_name": "approval_status_router"},
            "input_keys": ["approval_status"] # Router function will use this
            # Output keys usually not needed for a router itself, it directs flow.
        },
        "node_approved_action": {
            "id": "node_approved_action", "node_type": "agent_task", "name": "Perform Approved Action",
            "config": {"required_role": "ActionAgent", "task_description_template": "Execute action for approved request: {{user_request}}."},
            "input_keys": ["user_request", "approver_comments"], "output_keys": ["action_result"]
        },
        "node_rejected_action": {
            "id": "node_rejected_action", "node_type": "agent_task", "name": "Handle Rejection",
            "config": {"required_role": "NotificationAgent", "task_description_template": "Notify user of rejection for: {{user_request}}. Comments: {{approver_comments}}."},
            "input_keys": ["user_request", "approver_comments"], "output_keys": ["notification_status"]
        },
        "node_end_approved": {
            "id": "node_end_approved", "node_type": "end_node", "name": "End (Approved)"
        },
        "node_end_rejected": {
            "id": "node_end_rejected", "node_type": "end_node", "name": "End (Rejected)"
        }
    },
    "edges": [
        {"source_node_id": "node_start", "target_node_id": "node_get_approval"},
        {"source_node_id": "node_get_approval", "target_node_id": "node_approval_router"},
        # Edges from the router, distinguished by edge_label
        {"source_node_id": "node_approval_router", "target_node_id": "node_approved_action", "edge_label": "approved"},
        {"source_node_id": "node_approval_router", "target_node_id": "node_rejected_action", "edge_label": "rejected"},
        # Edges to end nodes
        {"source_node_id": "node_approved_action", "target_node_id": "node_end_approved"},
        {"source_node_id": "node_rejected_action", "target_node_id": "node_end_rejected"}
    ]
}

# workflow_graph = WorkflowGraph(**example_workflow_dict)
```

### Router Function Implementation

```python
# This function would be registered with WorkflowManager
def approval_status_router(global_state: Dict[str, Any]) -> str:
    approval_status = global_state.get("approval_status", "").lower()
    if approval_status == "approved":
        return "approved" # This string must match an edge_label from the router node
    elif approval_status == "rejected":
        return "rejected"
    else:
        return "error_unknown_status" # Or some default/error path
```

### Conceptual Execution

```python
# Assume workflow_manager is initialized and workflow loaded
# Assume approval_status_router is registered

# initial_data = {"user_request": "Please approve my vacation for next week."}
# final_state = await workflow_manager.process_workflow("simple_approval_workflow", initial_data)

# print(f"Workflow finished. Final State: {final_state}")
# If node_get_approval's mock output (or actual agent output) sets global_state["approval_status"] = "approved",
# the workflow would proceed to "node_approved_action".
```

## Future Enhancements

*   **Full AGENT_TASK/CREW_TASK Implementation**: Integrate `WorkflowManager` with `CrewManager` and individual agent execution logic to run actual agent/crew tasks within workflow nodes. This involves:
    *   Templating task descriptions for agents/crews using data from `global_state`.
    *   Managing the lifecycle of agent/crew tasks (e.g., creating `Task` objects for agents, `ExecutionPlan` for crews).
    *   Mapping results from `TaskResult.output` back to `global_state` based on `node.output_keys`.
*   **Data Transformation**: Allow defining transformations between `global_state` keys and node `input_keys`/`output_keys` if names don't match directly.
*   **Error Handling & Retries**: More sophisticated error handling strategies per node (e.g., retry logic, fallback paths not solely reliant on routers).
*   **Parallel Execution**: Support for executing independent branches of the workflow in parallel.
*   **Global State Schema Validation**: Enforce the `global_state_schema` if provided.
*   **Persistence**: Saving and loading workflow definitions from/to databases or files (e.g., YAML, JSON).
*   **Monitoring & Logging**: Enhanced logging and potentially a UI for visualizing workflow progress and state.
*   **More Node Types**: Introduce nodes for user input, external API calls (distinct from agent tools), loops, etc.
*   **Dynamic Task Generation**: Nodes that can dynamically generate further tasks or sub-workflows.
