from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum

class NodeType(str, Enum):
    """
    Defines the types of nodes that can exist in a workflow graph.
    """
    START = "start_node"
    END = "end_node"
    AGENT_TASK = "agent_task"  # Represents a task to be executed by a single agent
    CREW_TASK = "crew_task"    # Represents a more complex task/sub-plan to be executed by a CrewManager
    CONDITIONAL_ROUTER = "conditional_router_node" # Routes workflow based on conditions/state

class WorkflowNode(BaseModel):
    """
    Represents a single node in the workflow graph.
    """
    id: str = Field(description="Unique identifier for the node.")
    node_type: NodeType = Field(description="The type of the node.")
    name: str = Field(description="A human-readable name or label for the node.")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration specific to the node type. E.g., for AGENT_TASK, this might include agent_id or role_name, task_description template; for CONDITIONAL_ROUTER, the router_function_name."
    )
    input_keys: List[str] = Field(
        default_factory=list,
        description="List of keys expected from the global_state to be used as input for this node."
    )
    output_keys: List[str] = Field(
        default_factory=list,
        description="List of keys that this node is expected to produce and update in the global_state."
    )

class WorkflowEdge(BaseModel):
    """
    Represents a directed edge connecting two nodes in the workflow graph.
    """
    source_node_id: str = Field(description="The ID of the source node.")
    target_node_id: str = Field(description="The ID of the target node.")
    condition_name: Optional[str] = Field(
        default=None,
        description="Optional name of a condition. Used by CONDITIONAL_ROUTER nodes to decide the path. The router function returns a value that should match this condition_name (or an edge_label)."
    )
    edge_label: Optional[str] = Field(
        default=None,
        description="Optional label for the edge, typically used when a CONDITIONAL_ROUTER's output directly maps to an edge label to pick the next path."
    )
    # Note: A common pattern is for a router to return a string (e.g., "approved", "rejected")
    # This string can then be used to find an edge where `edge_label` matches this string.
    # `condition_name` might be used if the router checks a boolean condition by that name in global state,
    # but direct `edge_label` matching from router output is often more flexible.
    # For this initial design, `edge_label` is preferred for router outputs.

class WorkflowGraph(BaseModel):
    """
    Represents the entire workflow graph, including all nodes and edges.
    """
    graph_id: str = Field(description="Unique identifier for the workflow graph.")
    description: Optional[str] = Field(default=None, description="A brief description of the workflow's purpose.")
    nodes: Dict[str, WorkflowNode] = Field(
        default_factory=dict,
        description="A dictionary of nodes in the graph, keyed by their IDs."
    )
    edges: List[WorkflowEdge] = Field(
        default_factory=list,
        description="A list of edges defining the connections and flow between nodes."
    )
    start_node_id: Optional[str] = Field(
        default=None,
        description="The ID of the starting node for the workflow. Must be present in `nodes`."
    )
    # end_node_ids: List[str] = Field(default_factory=list, description="List of valid terminal node IDs.") # Multiple end nodes could exist
    global_state_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional JSON schema defining the expected structure and types of data in the global_state shared across the workflow."
    )

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        return self.nodes.get(node_id)

    def get_outgoing_edges(self, node_id: str) -> List[WorkflowEdge]:
        return [edge for edge in self.edges if edge.source_node_id == node_id]

    # Potential future validation:
    # - Ensure all node_ids in edges exist in nodes.
    # - Ensure start_node_id exists.
    # - Check for graph cycles (unless intentionally allowed for some router types).
    # - Validate that input_keys for a node can be satisfied by outputs of preceding nodes or initial_input.
