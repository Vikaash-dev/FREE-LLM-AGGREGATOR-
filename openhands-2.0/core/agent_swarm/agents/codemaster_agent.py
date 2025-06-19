import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CodeMasterAgent:
    """
    Advanced code generation and manipulation agent.
    Incorporates more detailed reasoning steps and self-reflection with action.
    """

    def __init__(self):
        self.name = "CodeMaster"
        self.capabilities = [
            'code_generation', 'code_optimization', 'code_refactoring', 'bug_fixing',
            'code_analysis', 'code_explanation', 'code_conversion',
            'pattern_recognition', 'syntax_validation', 'best_practices_application'
        ]
        self.supported_languages = [
            'python', 'javascript', 'typescript', 'java', 'c++',
            'go', 'rust', 'php', 'ruby', 'swift', 'kotlin'
        ]
        self.reflection_improvement_threshold = 0.75 # Example threshold

    async def initialize(self):
        logger.info(f"Initializing {self.name} Agent...")
        await asyncio.sleep(0.01)
        logger.info(f"{self.name} Agent initialized")

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.utcnow()
        final_agent_output: Dict[str, Any]
        try:
            task_type = self._determine_task_type(input_data)
            logger.info(f"{self.name} executing task type: {task_type} for input: {str(input_data)[:100]}")

            problem_representation = await self._create_problem_representation(input_data, task_type, context)
            logger.info(f"Problem representation for {task_type}: {str(problem_representation)[:100]}")

            initial_task_output: Dict[str, Any] # This is the direct output of the core task method
            if task_type == 'generate':
                initial_task_output = await self._generate_code(problem_representation, context)
            elif task_type == 'optimize':
                initial_task_output = await self._optimize_code(problem_representation, context)
            elif task_type == 'refactor':
                initial_task_output = await self._refactor_code(problem_representation, context)
            elif task_type == 'fix':
                initial_task_output = await self._fix_code(problem_representation, context)
            else:
                initial_task_output = await self._general_code_task(problem_representation, task_type, context)

            # `output_after_reflection` will contain `initial_task_output` possibly augmented with a `self_reflection` key.
            output_after_reflection = await self._self_reflect_on_output(task_type, initial_task_output, input_data, problem_representation)

            final_agent_output = output_after_reflection # Default to this if no improvement step is taken

            reflection_details = output_after_reflection.get('self_reflection', {})
            reflection_confidence = reflection_details.get('confidence', 1.0)
            improvement_suggestions = reflection_details.get('improvement_suggestions', [])

            # Check if improvement is warranted
            # Example: if confidence is low OR specific actionable suggestions exist
            needs_improvement = reflection_confidence < self.reflection_improvement_threshold
            if not needs_improvement and improvement_suggestions:
                # Mock check for "relevant" suggestions (e.g. not just generic ones)
                if any("alternative algorithms" in sugg.lower() or "re-verify" in sugg.lower() for sugg in improvement_suggestions):
                    needs_improvement = True

            if needs_improvement:
                logger.info(f"Reflection confidence ({reflection_confidence:.2f}) or suggestions warrant an improvement attempt for task {task_type}.")
                # Pass the direct output of the task method (initial_task_output) to the improvement method,
                # along with the reflection details derived from it.
                improved_output_package = await self._improve_solution_based_on_reflection(initial_task_output, reflection_details, context)

                if improved_output_package.get('improvement_applied_based_on_reflection'):
                    logger.info("Solution successfully improved based on reflection.")
                    # The improved_output_package is now the primary output.
                    # It should contain the improved task output and potentially its own reflection notes or a status.
                    final_agent_output = improved_output_package
                    # Ensure self_reflection from the improvement step is present or noted
                    if 'self_reflection' not in final_agent_output : final_agent_output['self_reflection'] = {}
                    final_agent_output['self_reflection']['status_after_improvement'] = 'successfully_improved'
                    final_agent_output['self_reflection']['original_reflection_confidence'] = reflection_confidence
                else:
                    logger.info("Improvement attempt based on reflection did not result in changes or was skipped.")
                    # Stick with output_after_reflection, but add a note about the attempt
                    if 'self_reflection' not in final_agent_output: final_agent_output['self_reflection'] = {}
                    final_agent_output['self_reflection']['status_after_improvement'] = 'improvement_attempted_no_change'
            else:
                logger.info(f"Reflection confidence ({reflection_confidence:.2f}) is sufficient, or no actionable suggestions. Using initial reflected output for task {task_type}.")


            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                'success': True,
                'agent': self.name,
                'task_type': task_type,
                'output': final_agent_output,
                'metadata': {
                    'execution_duration_seconds': execution_time,
                    'capabilities_used': self._get_used_capabilities(task_type),
                    'problem_representation_summary': str(problem_representation)[:100]
                }
            }

        except Exception as e:
            logger.exception(f"{self.name} execution error: {str(e)}")
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return {
                'success': False,
                'agent': self.name,
                'error': str(e),
                'metadata': {
                    'execution_duration_seconds': execution_time
                }
            }

    async def _improve_solution_based_on_reflection(self, original_task_output: Dict[str, Any], reflection_details: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates improving a solution based on self-reflection details."""
        logger.info(f"{self.name} attempting to improve solution based on reflection: {str(reflection_details.get('critique'))[:100]}...")
        await asyncio.sleep(0.02)

        # Make a deep copy to avoid modifying the original dict that might be used elsewhere
        # (e.g. if the original 'initial_task_output' was part of 'output_after_reflection' by reference)
        improved_output = {k: (v.copy() if isinstance(v, dict) else v) for k, v in original_task_output.items()}


        improvement_applied = False
        # Example: If critique mentioned efficiency for a code task
        if 'code' in improved_output and 'Efficiency could be improved' in reflection_details.get('critique', ''):
            improved_output['code'] = improved_output.get('code', '') + "\n// v2: Efficiency improvements based on reflection (simulated)."
            # Ensure reasoning_log is a list before appending
            if not isinstance(improved_output.get('reasoning_log'), list): improved_output['reasoning_log'] = []
            improved_output['reasoning_log'].append({'step': 'apply_reflection_efficiency_suggestion', 'status': 'completed', 'detail': 'Simulated efficiency enhancement.'})
            improvement_applied = True
            logger.info("Simulated efficiency improvement applied to code.")
        elif 'summary' in improved_output and 'not cover all aspects' in reflection_details.get('critique', ''): # Example for general tasks
            improved_output['summary'] = improved_output.get('summary', '') + " (v2: Expanded based on reflection to cover more aspects - simulated)."
            improvement_applied = True
            logger.info("Simulated content expansion based on reflection.")

        # Add a specific marker that this output is the result of an improvement iteration
        improved_output['improvement_applied_based_on_reflection'] = improvement_applied
        if not improvement_applied:
            logger.info("No specific simulated improvement action taken for this reflection, returning original task output structure.")
            # Return original structure but with the flag
            return original_task_output | {'improvement_applied_based_on_reflection': False}


        return improved_output

    async def _create_problem_representation(self, input_data: Dict[str, Any], task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.01)
        representation = {
            'original_input': input_data,
            'task_type_for_reasoning': task_type,
            'language': self._detect_language(input_data, context),
            'core_requirements': self._extract_requirements(input_data),
            'reasoning_steps_planned': [
                f'decompose_{task_type}_task',
                f'identify_key_{task_type}_components',
                f'draft_initial_{task_type}_solution_approach',
                f'refine_{task_type}_solution_details',
                f'verify_{task_type}_logic_coherence'
            ],
            'contextual_data': context
        }
        if 'code' in input_data:
            representation['existing_code_snippet'] = input_data['code'][:200]
        return representation

    async def _execute_reasoning_step(self, step_description: str, current_representation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Executing reasoning step: {step_description}")
        await asyncio.sleep(0.005)
        output_detail = f'Simulated output for {step_description}.'
        if 'identify' in step_description:
            output_detail += ' Identified components: Alpha, Beta, Gamma.'
        elif 'draft' in step_description:
            output_detail += ' Draft approach: Use Strategy Y.'
        return {'step': step_description, 'status': 'completed', 'output_detail': output_detail, 'timestamp': datetime.utcnow().isoformat()}

    async def _verify_reasoning_step(self, step_output: Dict[str, Any], criteria: List[str], context: Dict[str, Any]) -> bool:
        logger.debug(f"Verifying reasoning step output: {step_output.get('step')} against {criteria}")
        await asyncio.sleep(0.005)
        return True

    async def _process_reasoning_chain(self, representation: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        reasoning_log = []
        current_step_representation = representation.copy()
        for step_desc in representation.get('reasoning_steps_planned', []):
            step_output = await self._execute_reasoning_step(step_desc, current_step_representation, context)
            reasoning_log.append(step_output)
            is_verified = await self._verify_reasoning_step(step_output, ['coherence', 'logic'], context)
            step_output['verified'] = is_verified
            if not is_verified:
                logger.warning(f"Reasoning step {step_desc} failed verification.")
        return reasoning_log

    async def _generate_code(self, representation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        language = representation.get('language', 'python')
        requirements = representation.get('core_requirements', ['generic'])
        logger.info(f"Generating {language} code for: {requirements} based on reasoning plan.")
        reasoning_log = await self._process_reasoning_chain(representation, context)
        await asyncio.sleep(0.05)
        generated_code = f"// Simulated {language} code for {', '.join(requirements)}\n// Based on {len(reasoning_log)} reasoning steps.\nfunction reasonedExample() {{ return 'generated_after_reasoning'; }}"
        return {
            'code': generated_code, 'language': language, 'requirements_met': requirements,
            'validation': {'is_valid': True, 'errors': [], 'warnings': []},
            'best_practices_applied': True, 'documentation': f"Simulated documentation for {language} code.",
            'reasoning_log': reasoning_log
        }

    async def _optimize_code(self, representation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        existing_code = representation.get('existing_code_snippet', '// existing code')
        language = representation.get('language', 'python')
        reasoning_log = await self._process_reasoning_chain(representation, context)
        await asyncio.sleep(0.05)
        return {
            'original_code': existing_code, 'optimized_code': f"// Optimized {language} code\n{existing_code}", 'language': language,
            'analysis': {'complexity': 'O(n)', 'bottlenecks': ['sim_loop_inefficiency']},
            'improvements': {'performance_gain': '25%', 'memory_reduction': '10%'},
            'optimization_techniques': ['sim_loop_unrolling'], 'reasoning_log': reasoning_log
        }

    async def _refactor_code(self, representation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        existing_code = representation.get('existing_code_snippet', '// existing code for refactor')
        language = representation.get('language', 'python')
        reasoning_log = await self._process_reasoning_chain(representation, context)
        await asyncio.sleep(0.05)
        return {
            'original_code': existing_code, 'refactored_code': f"// Refactored {language} code\n{existing_code}", 'language': language,
            'structure_analysis': {'maintainability_score': 80, 'issues': ['sim_god_class']},
            'refactoring_patterns_applied': ['sim_extract_class'], 'validation': {'functionality_preserved': True, 'tests_pass': True},
            'reasoning_log': reasoning_log
        }

    async def _fix_code(self, representation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        buggy_code = representation.get('existing_code_snippet', '// buggy code')
        error_description = representation.get('original_input', {}).get('error', 'unknown error')
        language = representation.get('language', 'python')
        reasoning_log = await self._process_reasoning_chain(representation, context)
        await asyncio.sleep(0.05)
        return {
            'original_code': buggy_code, 'fixed_code': f"// Fixed {language} code for error: {error_description}\n{buggy_code}", 'language': language,
            'bug_analysis': {'root_cause': 'sim_off_by_one', 'severity': 'high'},
            'fixes_applied': ['sim_boundary_check'], 'validation': {'bugs_fixed': True, 'no_new_bugs': True},
            'reasoning_log': reasoning_log
        }

    async def _general_code_task(self, representation: Dict[str, Any], task_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        language = representation.get('language', 'python')
        code_snippet = representation.get('existing_code_snippet', '// code for general task')
        reasoning_log = await self._process_reasoning_chain(representation, context)
        await asyncio.sleep(0.05)
        output = {'task_performed': task_type, 'language': language, 'reasoning_log': reasoning_log}
        if task_type == 'review':
            output['review_summary'] = {'score': 8.5, 'comments': 'Simulated review: Looks good, consider adding more tests for edge cases.'}
        elif task_type == 'explain':
            output['explanation'] = f"This simulated {language} code, after reasoning, appears to implement X, Y, and Z."
        elif task_type == 'convert':
            output['target_language'] = representation.get('original_input',{}).get('target_language', 'javascript')
            output['converted_code'] = f"// Simulated conversion of {language} code to {output['target_language']} after reasoning."
        else: # analyze
            output['analysis_summary'] = f"Simulated analysis of {language} code completed after reasoning."
            output['details'] = {'lines': len(code_snippet.split('\n')), 'functions_found': 1 if 'function' in code_snippet else 0, 'reasoning_steps': len(reasoning_log)}
        return output

    async def _self_reflect_on_output(self, task_type: str, current_task_output: Dict[str, Any], input_data: Dict[str, Any], representation: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"{self.name} self-reflecting on {task_type} output.")
        await asyncio.sleep(0.01)
        mock_correctness_score = 0.9
        mock_efficiency_score = 0.6 # Lowered to trigger improvement example
        mock_clarity_score = 0.8
        mock_completeness_score = 0.95
        critique_parts = [f"Initial output for {task_type} by {self.name} processed."]
        improvement_suggestions = []
        if mock_correctness_score < 0.85: critique_parts.append("Potential correctness issues noted."); improvement_suggestions.append("Suggest re-verifying core logic.")
        if mock_efficiency_score < 0.75: critique_parts.append("Efficiency could be improved."); improvement_suggestions.append(f"Consider alternative algorithms for {representation.get('language', 'code')} optimization.")
        if mock_clarity_score < 0.75: critique_parts.append("Clarity could be enhanced."); improvement_suggestions.append("Add more detailed comments.")
        if mock_completeness_score < 0.9: critique_parts.append("Output may not cover all aspects."); improvement_suggestions.append("Review original requirements.")
        if not improvement_suggestions: critique_parts.append("Output seems reasonable."); improvement_suggestions.append("Consider alternative patterns.")
        reflection_summary = {
            'confidence': (mock_correctness_score + mock_efficiency_score + mock_clarity_score + mock_completeness_score) / 4.0,
            'critique': ' '.join(critique_parts), 'improvement_suggestions': improvement_suggestions,
            'evaluation_criteria_mock': {'correctness': mock_correctness_score, 'efficiency': mock_efficiency_score, 'clarity': mock_clarity_score, 'completeness': mock_completeness_score}
        }
        # Important: operate on a copy if current_task_output is to be preserved before reflection details are added
        output_with_reflection = current_task_output.copy()
        output_with_reflection['self_reflection'] = reflection_summary
        return output_with_reflection

    def _determine_task_type(self, input_data: Dict[str, Any]) -> str:
        text = input_data.get('text', '').lower()
        if any(word in text for word in ['generate', 'create', 'write', 'implement']): return 'generate'
        if any(word in text for word in ['optimize', 'improve performance', 'speed up']): return 'optimize'
        if any(word in text for word in ['refactor', 'restructure', 'clean up code']): return 'refactor'
        if any(word in text for word in ['fix', 'debug', 'resolve error', 'bug']): return 'fix'
        if any(word in text for word in ['review', 'code review']): return 'review'
        if any(word in text for word in ['explain', 'what does this code do']): return 'explain'
        if any(word in text for word in ['convert', 'translate code']): return 'convert'
        return 'analyze'

    def _extract_requirements(self, input_data: Dict[str, Any]) -> List[str]:
        text = input_data.get('text', '').lower(); reqs = []
        if 'function' in text: reqs.append('function_implementation')
        if 'class' in text: reqs.append('class_implementation')
        if 'api' in text: reqs.append('api_endpoint')
        if not reqs and text: reqs.append(text[:50])
        return reqs or ['generic_code_block']

    def _detect_language(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        text = input_data.get('text', '').lower(); code_snippet = input_data.get('code', '')
        for lang in self.supported_languages:
            if lang in text: return lang
        if 'language' in context and context['language'] in self.supported_languages: return context['language']
        if 'def ' in code_snippet and ':' in code_snippet : return 'python'
        if 'function ' in code_snippet and '{' in code_snippet and '}' in code_snippet: return 'javascript'
        if 'public static void main' in code_snippet: return 'java'
        if '#include' in code_snippet and 'std::' in code_snippet: return 'c++'
        if 'func ' in code_snippet and '{\n' in code_snippet: return 'go'
        if 'fn ' in code_snippet and '->' in code_snippet: return 'rust'
        return context.get('default_language', 'python')

    def _get_used_capabilities(self, task_type: str) -> List[str]:
        common_caps = ['pattern_recognition', 'syntax_validation']; task_specific_caps = {
            'generate': ['code_generation', 'best_practices_application'], 'optimize': ['code_optimization'],
            'refactor': ['code_optimization', 'best_practices_application'], 'fix': [],
            'review': ['code_analysis'], 'explain': ['code_analysis'],
            'convert': ['code_analysis', 'code_generation'], 'analyze': ['code_analysis']}
        return list(set(common_caps + task_specific_caps.get(task_type, ['code_analysis'])))

__all__ = ['CodeMasterAgent']
