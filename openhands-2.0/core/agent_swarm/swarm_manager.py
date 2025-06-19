import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import uuid

# Placeholder imports for agents
try:
    from .agents.codemaster_agent import CodeMasterAgent
    # Import other agents if they have specific types needed here, otherwise BaseAgent is enough for placeholders
    from .agents.architect_agent import ArchitectAgent
    from .agents.security_agent import SecurityAgent
    from .agents.test_agent import TestAgent
    from .agents.refactor_agent import RefactorAgent
    from .agents.document_agent import DocumentAgent
    from .agents.deploy_agent import DeployAgent
    from .agents.research_agent import ResearchAgent
except ImportError:
    class BaseAgent:
        def __init__(self, name="BaseAgent"): # Added name init
            self.name = name
        async def initialize(self): logger.info(f"Initializing {self.name}...")
        async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            logger.info(f"{self.name} executing with {input_data}")
            await asyncio.sleep(0.01)
            return {'success': True, 'output': {'message': f'{self.name} executed'}, 'agent': self.name}
    class CodeMasterAgent(BaseAgent): pass; class ArchitectAgent(BaseAgent): pass; class SecurityAgent(BaseAgent): pass; class TestAgent(BaseAgent): pass; class RefactorAgent(BaseAgent): pass; class DocumentAgent(BaseAgent): pass; class DeployAgent(BaseAgent): pass; class ResearchAgent(BaseAgent): pass

# Placeholder for AgentConfiguration if not imported
if 'AgentConfiguration' not in globals(): # Check if AgentConfiguration is already defined
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
        # Removed PipelineExecutor and ParallelProcessor initializations as they are not used with new plan based exec.
        self.active_tasks: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}

    async def initialize(self):
        logger.info("Initializing Agent Swarm...")
        self.agents = {
            'codemaster': CodeMasterAgent(name='CodeMasterAgent'), 'architect': ArchitectAgent(name='ArchitectAgent'),
            'security': SecurityAgent(name='SecurityAgent'), 'test': TestAgent(name='TestAgent'),
            'refactor': RefactorAgent(name='RefactorAgent'), 'document': DocumentAgent(name='DocumentAgent'),
            'deploy': DeployAgent(name='DeployAgent'), 'research': ResearchAgent(name='ResearchAgent')
        } # Pass names to BaseAgent constructor if using it
        await asyncio.gather(*[agent.initialize() for agent in self.agents.values()])
        # Removed init for PipelineExecutor and ParallelProcessor
        logger.info("Agent Swarm initialized successfully")

    async def _create_internal_execution_plan_for_agents(
        self, config: AgentConfiguration, input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Creates a sequence of agent execution steps based on the config."""
        internal_plan: List[Dict[str, Any]] = []
        agent_execution_order = []
        if config.priority_order:
            valid_priority_agents = [agent for agent in config.priority_order if agent in config.selected_agents]
            if valid_priority_agents:
                agent_execution_order = valid_priority_agents
                for agent in config.selected_agents:
                    if agent not in agent_execution_order:
                        agent_execution_order.append(agent)
            else:
                agent_execution_order = config.selected_agents
        else:
            agent_execution_order = config.selected_agents

        for i, agent_name in enumerate(agent_execution_order):
            if agent_name not in self.agents:
                logger.warning(f"Agent '{agent_name}' specified in plan but not found in initialized agents. Skipping.")
                continue
            internal_plan.append({
                'step_id': i + 1,
                'agent_name': agent_name,
                'sub_task_input': input_data.copy(),
                'status': 'pending'
            })
        logger.info(f"Generated internal execution plan with {len(internal_plan)} steps: {[s['agent_name'] for s in internal_plan]}")
        return internal_plan

    async def select_optimal_agents(self, task_analysis: Dict[str, Any]) -> AgentConfiguration:
        required_agents = task_analysis.get('recommended_agents', ['codemaster'])
        complexity = task_analysis.get('overall_complexity', 0.5)
        collaboration_pattern = self._determine_collaboration_pattern(required_agents, complexity)
        execution_strategy = self._choose_execution_strategy(required_agents, task_analysis)
        resource_allocation = self._allocate_resources(required_agents, task_analysis)
        priority_order = self._determine_priority_order(required_agents, task_analysis)
        return AgentConfiguration(selected_agents=required_agents, collaboration_pattern=collaboration_pattern, execution_strategy=execution_strategy, resource_allocation=resource_allocation, priority_order=priority_order)

    async def execute_task(self, config: AgentConfiguration, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        task_id = self._generate_task_id()
        start_time = datetime.utcnow()
        try:
            logger.info(f"Executing task {task_id} with {len(config.selected_agents)} agents using {config.execution_strategy} strategy")
            self.active_tasks[task_id] = {'config': config, 'start_time': start_time, 'status': 'running'}

            internal_plan = await self._create_internal_execution_plan_for_agents(config, input_data, context)
            if not internal_plan:
                logger.warning(f"Task {task_id}: No valid agents found in plan. Aborting execution.")
                return {'success': False, 'error': 'No valid agents to execute plan.', 'task_id': task_id}

            result: Dict[str, Any] = {}
            if config.execution_strategy == 'pipeline':
                result = await self._execute_pipeline(internal_plan, context, task_id, config)
            elif config.execution_strategy == 'parallel':
                result = await self._execute_parallel(internal_plan, context, task_id, config)
            elif config.execution_strategy == 'hybrid':
                result = await self._execute_hybrid(internal_plan, context, task_id, config)
            else:
                result = await self._execute_sequential(internal_plan, context, task_id, config)

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

    async def _execute_pipeline(self, internal_plan: List[Dict[str, Any]], context: Dict[str, Any], task_id: str, config: AgentConfiguration) -> Dict[str, Any]:
        current_step_data: Dict[str, Any] = {}
        step_results: Dict[str, Any] = {}

        for step_info in internal_plan:
            agent_name = step_info['agent_name']
            sub_task_input = step_info['sub_task_input']

            current_input_for_agent = sub_task_input.copy()
            if current_step_data: # Output from previous step can be merged into input for the next
                current_input_for_agent.update(current_step_data)

            agent = self.agents[agent_name]
            logger.info(f"Task {task_id} (Pipeline Step {step_info['step_id']}): Executing {agent_name}")
            agent_result = await agent.execute(current_input_for_agent, context.copy())
            step_results[f"step_{step_info['step_id']}_{agent_name}"] = agent_result

            if agent_result.get('success', False):
                current_step_data = agent_result.get('output', {})
                context.update(agent_result.get('context_updates', {})) # Allow agents to update shared context
            else:
                logger.warning(f"Agent {agent_name} in pipeline step {step_info['step_id']} failed. Halting pipeline.")
                break
        return self._synthesize_agent_results(step_results, config, 'pipeline', internal_plan)

    async def _execute_parallel(self, internal_plan: List[Dict[str, Any]], context: Dict[str, Any], task_id: str, config: AgentConfiguration) -> Dict[str, Any]:
        logger.info(f"Task {task_id} (Parallel): Executing {len(internal_plan)} agent steps")
        tasks_to_run = []
        agent_step_names_in_plan = [] # To map results back correctly
        for step_info in internal_plan:
            agent_name = step_info['agent_name']
            sub_task_input = step_info['sub_task_input']
            agent = self.agents[agent_name]
            tasks_to_run.append(asyncio.create_task(agent.execute(sub_task_input.copy(), context.copy()), name=f"{task_id}_step_{step_info['step_id']}_{agent_name}"))
            agent_step_names_in_plan.append(f"step_{step_info['step_id']}_{agent_name}")

        completed_tasks_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
        step_results: Dict[str, Any] = {}
        for i, agent_step_name in enumerate(agent_step_names_in_plan):
            if isinstance(completed_tasks_results[i], Exception):
                logger.error(f"Agent step {agent_step_name} raised an exception: {completed_tasks_results[i]}")
                step_results[agent_step_name] = {'success': False, 'error': str(completed_tasks_results[i])}
            else:
                step_results[agent_step_name] = completed_tasks_results[i]
        return self._synthesize_agent_results(step_results, config, 'parallel', internal_plan)

    async def _execute_hybrid(self, internal_plan: List[Dict[str, Any]], context: Dict[str, Any], task_id: str, config: AgentConfiguration) -> Dict[str, Any]:
        logger.info(f"Task {task_id} (Hybrid): Simplified to pipeline execution for now.")
        return await self._execute_pipeline(internal_plan, context, task_id, config)

    async def _execute_sequential(self, internal_plan: List[Dict[str, Any]], context: Dict[str, Any], task_id: str, config: AgentConfiguration) -> Dict[str, Any]:
        logger.info(f"Task {task_id} (Sequential): Executing agent steps sequentially.")
        return await self._execute_pipeline(internal_plan, context, task_id, config)

    def _determine_collaboration_pattern(self, agents: List[str], complexity: float) -> str:
        if len(agents) == 1: return 'single';
        if complexity > 0.7: return 'hierarchical';
        return 'swarm'

    def _choose_execution_strategy(self, agents: List[str], task_analysis: Dict[str, Any]) -> str:
        if len(agents) == 1: return 'sequential'
        if task_analysis.get('requires_planning', False) or len(task_analysis.get('dependencies',[])) > 0:
            return 'pipeline'
        if task_analysis.get('overall_complexity', 0.5) > 0.8 and len(agents) > 2: return 'hybrid'
        if len(agents) <= 3 : return 'parallel'
        return 'pipeline'

    def _allocate_resources(self, agents: List[str], task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        total_resources = task_analysis.get('resource_requirements', {})
        agent_count = len(agents) if agents else 1
        # Ensure agent_count is not zero to prevent ZeroDivisionError
        if agent_count == 0: agent_count = 1
        base_alloc = {'memory_mb': total_resources.get('memory_mb', 512)//agent_count, 'cpu_cores': total_resources.get('cpu_cores',1.0)/agent_count, 'timeout_seconds': 300}
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

        # Log results based on the actual execution plan steps for clarity
        for step_info in internal_plan:
            step_key = f"step_{step_info['step_id']}_{step_info['agent_name']}"
            result_data = step_results.get(step_key)

            if isinstance(result_data, dict) and result_data.get('success', False):
                # More sophisticated merge logic might be needed depending on agent outputs
                # For now, a simple update, prioritizing later steps for overlapping keys
                final_combined_output.update(result_data.get('output', {}))
            elif isinstance(result_data, dict): # Step failed
                all_steps_successful = False
                # Store specific error from the failed step
                final_combined_output[f"{step_key}_error"] = result_data.get('error', 'Unknown error in step')
                if strategy == 'pipeline': # If pipeline, failure of one step implies overall failure
                    logger.warning(f"Pipeline strategy: Step {step_key} failed. Overall success set to False.")
                    # No need to break here as we are just processing results, execution already stopped if needed
            else: # Unexpected result format
                all_steps_successful = False
                final_combined_output[f"{step_key}_error"] = "Invalid result format from step"
                if strategy == 'pipeline':
                     logger.warning(f"Pipeline strategy: Step {step_key} had invalid result. Overall success set to False.")

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
