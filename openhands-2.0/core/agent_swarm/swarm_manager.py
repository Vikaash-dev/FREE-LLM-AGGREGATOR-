import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import uuid

# Placeholder imports for agents
try:
    from .agents.codemaster_agent import CodeMasterAgent
    from .agents.architect_agent import ArchitectAgent
    from .agents.security_agent import SecurityAgent
    from .agents.test_agent import TestAgent
    from .agents.refactor_agent import RefactorAgent
    from .agents.document_agent import DocumentAgent
    from .agents.deploy_agent import DeployAgent
    from .agents.research_agent import ResearchAgent
except ImportError:
    class BaseAgent:
        def __init__(self, name="BaseAgent"):
            self.name = name
        async def initialize(self): logger.info(f"Initializing {self.name}...")
        async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"{self.name} executing task {input_data.get('description','unspecified')} with context {str(context)[:50]}")
            await asyncio.sleep(0.01) # Simulate work
            return {'success': True, 'output': {f'{self.name}_result': 'completed'}, 'agent': self.name}
    class CodeMasterAgent(BaseAgent): pass # type: ignore
    class ArchitectAgent(BaseAgent): pass # type: ignore
    class SecurityAgent(BaseAgent): pass # type: ignore
    class TestAgent(BaseAgent): pass # type: ignore
    class RefactorAgent(BaseAgent): pass # type: ignore
    class DocumentAgent(BaseAgent): pass # type: ignore
    class DeployAgent(BaseAgent): pass # type: ignore
    class ResearchAgent(BaseAgent): pass # type: ignore

# Ensure AgentConfiguration is defined or imported
if 'AgentConfiguration' not in globals():
    @dataclass
    class AgentConfiguration:
        selected_agents: List[str]
        collaboration_pattern: str
        execution_strategy: str
        resource_allocation: Dict[str, Any]
        priority_order: List[str] = field(default_factory=list)

logger = logging.getLogger(__name__)

class AgentSwarmManager:
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.active_tasks: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}

    async def initialize(self):
        logger.info("Initializing Agent Swarm...")
        self.agents = {
            'codemaster': CodeMasterAgent(name="CodeMasterAgent"),
            'architect': ArchitectAgent(name="ArchitectAgent"),
            'security': SecurityAgent(name="SecurityAgent"),
            'test': TestAgent(name="TestAgent"),
            'refactor': RefactorAgent(name="RefactorAgent"),
            'document': DocumentAgent(name="DocumentAgent"),
            'deploy': DeployAgent(name="DeployAgent"),
            'research': ResearchAgent(name="ResearchAgent")
        }
        # Initialize all agents concurrently
        await asyncio.gather(*[agent.initialize() for agent in self.agents.values()])
        logger.info("Agent Swarm initialized successfully")

    async def _create_internal_execution_plan_for_agents(
        self, config: AgentConfiguration, input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Creates a sequence of agent execution steps based on the config."""
        internal_plan: List[Dict[str, Any]] = []
        agent_execution_order = config.priority_order if config.priority_order else config.selected_agents

        valid_execution_order = [agent_name for agent_name in agent_execution_order if agent_name in config.selected_agents and agent_name in self.agents]

        if not valid_execution_order:
            logger.warning("No valid agents found in execution order based on selected_agents and priority_order.")
            return []

        # For pipeline, the input_data for the first step is the initial input_data.
        # For subsequent steps, the sub_task_input will be the output of the previous step.
        # This method just outlines the sequence; actual input passing is handled by _execute_pipeline.
        for i, agent_name in enumerate(valid_execution_order):
            internal_plan.append({
                'step_id': i + 1,
                'agent_name': agent_name,
                # For pipeline, initial input to the step is dynamic. For parallel, it's often the same.
                'sub_task_input': input_data.copy() if config.execution_strategy == 'parallel' else {},
                'status': 'pending'
            })
        logger.info(f"Generated internal execution plan: {[(s['step_id'], s['agent_name']) for s in internal_plan]}")
        return internal_plan

    async def select_optimal_agents(self, task_analysis: Dict[str, Any]) -> AgentConfiguration:
        required_agents = task_analysis.get('recommended_agents', ['codemaster'])
        available_and_required = [agent for agent in required_agents if agent in self.agents]
        if not available_and_required:
            logger.warning(f"No available agents for recommended: {required_agents}. Defaulting to codemaster.")
            available_and_required = ['codemaster'] if 'codemaster' in self.agents else []
            if not available_and_required and self.agents:
                 available_and_required = [next(iter(self.agents.keys()))]
                 logger.warning(f"CodeMaster not available, defaulting to {available_and_required[0]}")
            elif not self.agents:
                logger.error("No agents initialized in AgentSwarmManager!")
                return AgentConfiguration([], '', '', {}, [])

        complexity = task_analysis.get('overall_complexity', 0.5)
        collaboration_pattern = self._determine_collaboration_pattern(available_and_required, complexity)
        execution_strategy = self._choose_execution_strategy(available_and_required, task_analysis)
        resource_allocation = self._allocate_resources(available_and_required, task_analysis)
        priority_order = self._determine_priority_order(available_and_required, task_analysis)

        return AgentConfiguration(
            selected_agents=available_and_required,
            collaboration_pattern=collaboration_pattern,
            execution_strategy=execution_strategy,
            resource_allocation=resource_allocation,
            priority_order=priority_order
        )

    async def execute_task(self, config: AgentConfiguration, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        task_id = self._generate_task_id()
        start_time = datetime.utcnow()
        try:
            logger.info(f"Executing task {task_id} with {len(config.selected_agents)} agents using '{config.execution_strategy}' strategy.")
            self.active_tasks[task_id] = {'config': config, 'start_time': start_time, 'status': 'running'}

            # Pass the original input_data to _create_internal_execution_plan_for_agents.
            # The plan will then determine what input each agent step gets, especially for parallel.
            internal_plan = await self._create_internal_execution_plan_for_agents(config, input_data, context)
            if not internal_plan:
                logger.warning(f"Task {task_id}: No valid agent execution plan generated. Aborting execution.")
                return {'success': False, 'error': 'Failed to create a valid execution plan for agents.', 'task_id': task_id}

            result: Dict[str, Any] = {}
            # Pass initial_input_data to execution strategies for the first step or parallel distribution
            if config.execution_strategy == 'pipeline':
                result = await self._execute_pipeline(internal_plan, input_data, context, task_id, config)
            elif config.execution_strategy == 'parallel':
                result = await self._execute_parallel(internal_plan, input_data, context, task_id, config)
            elif config.execution_strategy == 'hybrid':
                result = await self._execute_hybrid(internal_plan, input_data, context, task_id, config)
            else:
                logger.warning(f"Execution strategy '{config.execution_strategy}' not specifically implemented or recognized, defaulting to pipeline.")
                result = await self._execute_pipeline(internal_plan, input_data, context, task_id, config)

            self.active_tasks[task_id]['status'] = 'completed'
            self.active_tasks[task_id]['result'] = result
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self._record_performance_metrics(task_id, config, execution_time, result)
            return result
        except Exception as e:
            logger.exception(f"Error executing task {task_id}: {str(e)}")
            if task_id in self.active_tasks:
                 self.active_tasks[task_id]['status'] = 'failed'; self.active_tasks[task_id]['error'] = str(e)
            return {'success': False, 'error': str(e), 'task_id': task_id}
        finally:
            if task_id in self.active_tasks: self.active_tasks[task_id]['end_time'] = datetime.utcnow()

    async def _execute_pipeline(self, internal_plan: List[Dict[str, Any]], initial_input_data: Dict[str, Any], context: Dict[str, Any], task_id: str, config: AgentConfiguration) -> Dict[str, Any]:
        current_step_input_data = initial_input_data.copy() # Start with the overall task input for the first agent
        step_results: Dict[str, Any] = {}

        for step_info in internal_plan:
            agent_name = step_info['agent_name']
            agent = self.agents[agent_name]
            logger.info(f"Task {task_id} (Pipeline Step {step_info['step_id']}: {agent_name}): Executing.")

            # The 'current_step_input_data' carries the output from the previous step.
            agent_result = await agent.execute(current_step_input_data, context.copy())
            step_results[f"step_{step_info['step_id']}_{agent_name}"] = agent_result

            if agent_result.get('success', False):
                current_step_input_data = agent_result.get('output', {}) # Output of this step is input for the next
                if not isinstance(current_step_input_data, dict): # Ensure it's a dict for the next agent
                    logger.warning(f"Output from agent {agent_name} is not a dict, wrapping it. Output: {str(current_step_input_data)[:100]}")
                    current_step_input_data = {'previous_agent_output': current_step_input_data}
                context.update(agent_result.get('context_updates', {}))
            else:
                logger.warning(f"Agent {agent_name} in pipeline step {step_info['step_id']} failed. Halting pipeline.")
                break
        return self._synthesize_agent_results(step_results, config, 'pipeline', internal_plan)

    async def _execute_parallel(self, internal_plan: List[Dict[str, Any]], initial_input_data: Dict[str, Any], context: Dict[str, Any], task_id: str, config: AgentConfiguration) -> Dict[str, Any]:
        logger.info(f"Task {task_id} (Parallel): Executing {len(internal_plan)} agent steps in parallel.")
        aws_tasks = []
        agent_step_names = []
        for step_info in internal_plan:
            agent_name = step_info['agent_name']
            # In parallel, each agent typically gets the initial_input_data unless plan specifies otherwise for sub_task_input
            sub_task_input_for_agent = step_info['sub_task_input'] if step_info['sub_task_input'] else initial_input_data
            agent = self.agents[agent_name]
            aws_tasks.append(agent.execute(sub_task_input_for_agent.copy(), context.copy()))
            agent_step_names.append(f"step_{step_info['step_id']}_{agent_name}")

        results_from_gather = await asyncio.gather(*aws_tasks, return_exceptions=True)

        step_results: Dict[str, Any] = {}
        for i, agent_step_name in enumerate(agent_step_names):
            if isinstance(results_from_gather[i], Exception):
                logger.error(f"Agent step {agent_step_name} raised an exception: {results_from_gather[i]}")
                step_results[agent_step_name] = {'success': False, 'error': str(results_from_gather[i])}
            else:
                step_results[agent_step_name] = results_from_gather[i]
        return self._synthesize_agent_results(step_results, config, 'parallel', internal_plan)

    async def _execute_hybrid(self, internal_plan: List[Dict[str, Any]], initial_input_data: Dict[str, Any], context: Dict[str, Any], task_id: str, config: AgentConfiguration) -> Dict[str, Any]:
        logger.info(f"Task {task_id} (Hybrid): Simplified to pipeline execution for now.")
        return await self._execute_pipeline(internal_plan, initial_input_data, context, task_id, config)

    async def _execute_sequential(self, internal_plan: List[Dict[str, Any]], initial_input_data: Dict[str, Any], context: Dict[str, Any], task_id: str, config: AgentConfiguration) -> Dict[str, Any]:
        logger.info(f"Task {task_id} (Sequential): Executing agent steps sequentially.")
        return await self._execute_pipeline(internal_plan, initial_input_data, context, task_id, config)

    def _determine_collaboration_pattern(self, agents: List[str], complexity: float) -> str:
        if not agents: return 'none'
        if len(agents) == 1: return 'single'
        if complexity > 0.7: return 'hierarchical'
        return 'swarm'

    def _choose_execution_strategy(self, agents: List[str], task_analysis: Dict[str, Any]) -> str:
        if not agents: return 'none'
        if len(agents) == 1: return 'sequential'
        if task_analysis.get('requires_planning', False) or len(task_analysis.get('dependencies',[])) > 0:
            return 'pipeline'
        if task_analysis.get('overall_complexity', 0.5) > 0.8 and len(agents) > 2: return 'hybrid'
        if len(agents) <= 3 : return 'parallel'
        return 'pipeline'

    def _allocate_resources(self, agents: List[str], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        if not agents: return {}
        total_resources = task_analysis.get('resource_requirements', {})
        agent_count = len(agents)
        base_alloc = {
            'memory_mb': total_resources.get('memory_mb', 512) // agent_count if agent_count > 0 else 512,
            'cpu_cores': total_resources.get('cpu_cores', 1.0) / agent_count if agent_count > 0 else 1.0,
            'timeout_seconds': 300
        }
        return {agent_name: base_alloc.copy() for agent_name in agents}

    def _determine_priority_order(self, agents: List[str], task_analysis: Dict[str, Any]) -> List[str]:
        priority_map = {'research':1, 'architect':2, 'security':3, 'codemaster':4, 'test':5, 'refactor':6, 'document':7, 'deploy':8}
        present_agents = [agent for agent in agents if agent in priority_map]
        sorted_present_agents = sorted(present_agents, key=lambda x: priority_map[x])
        other_agents = [agent for agent in agents if agent not in priority_map]
        return sorted_present_agents + other_agents

    def _synthesize_agent_results(self, step_results: Dict[str, Any], config: AgentConfiguration, strategy: str, internal_plan: List[Dict[str,Any]]) -> Dict[str, Any]:
        final_combined_output: Dict[str, Any] = {}
        all_steps_successful = True

        for step_info in internal_plan:
            step_key = f"step_{step_info['step_id']}_{step_info['agent_name']}"
            result_data = step_results.get(step_key)
            if isinstance(result_data, dict):
                if result_data.get('success', False):
                    agent_specific_output = result_data.get('output', {})
                    # Prefix keys from agent output to avoid collision, e.g. {'CodeMasterAgent_code': '...', 'ArchitectAgent_plan': '...'}
                    # Or define a more structured way to merge, e.g. only specific keys are promoted.
                    for k, v in agent_specific_output.items():
                        final_combined_output[f"{step_info['agent_name']}_{k}"] = v
                else:
                    all_steps_successful = False
                    final_combined_output[f"{step_key}_error"] = result_data.get('error', 'Unknown error in step')
            elif result_data is not None:
                all_steps_successful = False
                final_combined_output[f"{step_key}_error"] = "Invalid result format from step"
            # If a step was planned but has no result (e.g. agent skipped), it's a form of failure for strict pipelines
            elif strategy == 'pipeline' and result_data is None:
                all_steps_successful = False
                final_combined_output[f"{step_key}_error"] = "Agent skipped or no result produced."


        if all_steps_successful and not final_combined_output and step_results:
            final_combined_output['message'] = 'Agents executed successfully but produced no combined output.'
        elif not final_combined_output and not all_steps_successful:
             final_combined_output['message'] = 'Execution failed and no output was generated.'


        return {
            'success': all_steps_successful,
            'execution_strategy': strategy,
            'step_results': step_results,
            'synthesized_output': final_combined_output,
            'metadata': {'agents_in_plan': [s['agent_name'] for s in internal_plan], 'total_steps': len(internal_plan)}
        }

    async def _record_performance_metrics(self, task_id: str, config: AgentConfiguration, execution_time: float, result: Dict[str, Any]):
        metrics = {'task_id':task_id, 'timestamp':datetime.utcnow(), 'execution_time':execution_time, 'agents_used':config.selected_agents, 'execution_strategy':config.execution_strategy, 'success':result.get('success',False), 'agent_count':len(config.selected_agents)}
        self.performance_metrics[task_id] = metrics
        logger.info(f"Task {task_id} metrics recorded. Time: {execution_time:.2f}s, Success: {result.get('success')}")

    def _generate_task_id(self) -> str:
        return f"task_{uuid.uuid4().hex[:8]}"

__all__ = ['AgentSwarmManager', 'AgentConfiguration']
