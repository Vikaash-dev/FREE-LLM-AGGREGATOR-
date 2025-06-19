import re
import asyncio # Added though not explicitly used in the snippet, good for consistency
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RESEARCH_INTEGRATION = "research_integration"
    # New/Refined Task Types
    COMPLEX_REASONING = "complex_reasoning" # For tasks requiring multi-step thought
    KNOWLEDGE_LOOKUP = "knowledge_lookup"   # For tasks requiring factual/external data
    PLANNING = "planning"                 # For tasks that need a sequence of actions defined
    UNKNOWN = "unknown"

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"
    # Potentially add more granular complexities if needed later
    REQUIRES_DEEP_REASONING = "requires_deep_reasoning"
    REQUIRES_KNOWLEDGE_GRAPH = "requires_knowledge_graph"

@dataclass
class TaskAnalysis:
    task_type: TaskType
    complexity: TaskComplexity # This could become a list or a more complex object if needed
    estimated_duration: int # in seconds
    required_agents: List[str]
    dependencies: List[str]
    risk_level: float # 0.0 to 1.0
    confidence: float # 0.0 to 1.0
    metadata: Dict[str, Any]
    # Add new fields to flag specific needs
    requires_advanced_reasoning: bool = False
    requires_external_knowledge: bool = False
    requires_planning: bool = False

class TaskAnalyzer:
    """
    Advanced task analysis system for intelligent routing
    """

    def __init__(self):
        self.task_patterns = self._initialize_patterns()
        self.complexity_indicators = self._initialize_complexity_indicators()

    def _initialize_patterns(self) -> Dict[TaskType, List[str]]:
        patterns = {
            TaskType.CODE_GENERATION: [
                r'create\s+(?:a\s+)?(?:new\s+)?(?:function|class|module|component)',
                r'generate\s+(?:code|function|class)',
                r'implement\s+(?:a\s+)?(?:function|algorithm|feature)',
                r'write\s+(?:a\s+)?(?:function|class|script)',
                r'build\s+(?:a\s+)?(?:component|module|system)'
            ],
            TaskType.BUG_FIX: [
                r'fix\s+(?:the\s+)?(?:bug|error|issue|problem)',
                r'debug\s+(?:the\s+)?(?:code|function|issue)',
                r'resolve\s+(?:the\s+)?(?:error|issue|problem)',
                r'correct\s+(?:the\s+)?(?:bug|mistake|error)',
                r'repair\s+(?:the\s+)?(?:broken|faulty)'
            ],
            TaskType.REFACTORING: [
                r'refactor\s+(?:the\s+)?(?:code|function|class)',
                r'improve\s+(?:the\s+)?(?:code|structure|design)',
                r'optimize\s+(?:the\s+)?(?:code|performance|structure)',
                r'clean\s+up\s+(?:the\s+)?code',
                r'restructure\s+(?:the\s+)?(?:code|project)'
            ],
            TaskType.TESTING: [
                r'test\s+(?:the\s+)?(?:code|function|component)',
                r'write\s+(?:unit\s+)?tests',
                r'create\s+test\s+cases',
                r'add\s+testing',
                r'verify\s+(?:the\s+)?functionality'
            ],
            TaskType.DOCUMENTATION: [
                r'document\s+(?:the\s+)?(?:code|function|api)',
                r'write\s+documentation',
                r'create\s+(?:api\s+)?docs',
                r'add\s+comments',
                r'explain\s+(?:the\s+)?code'
            ],
            TaskType.ARCHITECTURE: [
                r'design\s+(?:the\s+)?(?:architecture|system|structure)',
                r'architect\s+(?:a\s+)?(?:system|solution)',
                # r'plan\s+(?:the\s+)?(?:system|architecture)', # Moved to PLANNING
                r'structure\s+(?:the\s+)?project',
                r'design\s+patterns'
            ],
            TaskType.SECURITY_AUDIT: [
                r'security\s+(?:audit|review|check)',
                r'find\s+(?:security\s+)?vulnerabilities',
                r'check\s+for\s+(?:security\s+)?issues',
                r'audit\s+(?:the\s+)?(?:code|security)',
                r'vulnerability\s+assessment'
            ],
            TaskType.PERFORMANCE_OPTIMIZATION: [
                r'optimize\s+(?:the\s+)?performance',
                r'improve\s+(?:the\s+)?speed',
                r'make\s+(?:it\s+)?faster',
                r'performance\s+tuning',
                r'speed\s+up\s+(?:the\s+)?(?:code|application)'
            ],
            TaskType.RESEARCH_INTEGRATION: [
                r'integrate\s+(?:latest\s+)?research',
                r'implement\s+(?:new\s+)?(?:algorithm|technique)',
                r'use\s+(?:latest\s+)?(?:ai|ml|research)',
                r'apply\s+(?:new\s+)?(?:methods|techniques)',
                r'cutting[- ]edge\s+(?:approach|method)'
            ],
            TaskType.COMPLEX_REASONING: [
                r'explain\s+step-by-step', r'deduce', r'what if', r'reason about',
                r'analyze implications', r'logical conclusion', r'problem solve'
            ],
            TaskType.KNOWLEDGE_LOOKUP: [
                r'find information about', r'research topic', r'get facts on',
                r'lookup', r'tell me about', r'what is known about'
            ],
            TaskType.PLANNING: [
                r'plan\s+for', r'create a plan', r'outline steps',
                r'sequence of actions', r'develop a strategy for', r'roadmap for'
            ]
        }
        return patterns

    def _initialize_complexity_indicators(self) -> Dict[str, float]:
        # Existing indicators are fine, no change needed here for now
        return {
            'simple': 0.1, 'basic': 0.1, 'easy': 0.1, 'quick': 0.2,
            'small': 0.2, 'minor': 0.2, 'trivial': 0.1,
            'moderate': 0.4, 'medium': 0.4, 'standard': 0.4,
            'typical': 0.4, 'normal': 0.4, 'regular': 0.4,
            'complex': 0.7, 'advanced': 0.7, 'sophisticated': 0.8,
            'comprehensive': 0.8, 'detailed': 0.6, 'thorough': 0.7,
            'expert': 0.9, 'cutting-edge': 1.0, 'state-of-the-art': 1.0,
            'revolutionary': 1.0, 'innovative': 0.9, 'novel': 0.9,
            'research-grade': 1.0, 'enterprise-grade': 0.8
        }

    async def analyze_task(self, input_text: str, context: Dict[str, Any] = None) -> TaskAnalysis:
        context = context or {}
        task_type = await self._detect_task_type(input_text)

        # Determine flags for advanced capabilities BEFORE general complexity
        requires_advanced_reasoning = task_type == TaskType.COMPLEX_REASONING or \
                                     any(kw in input_text.lower() for kw in ['think step-by-step', 'deliberate thought'])
        requires_external_knowledge = task_type == TaskType.KNOWLEDGE_LOOKUP or \
                                      any(kw in input_text.lower() for kw in ['search web for', 'find recent data on'])
        requires_planning = task_type == TaskType.PLANNING

        complexity_enum, complexity_score = await self._analyze_complexity(input_text, task_type, requires_advanced_reasoning, requires_external_knowledge)

        duration = await self._estimate_duration(input_text, task_type, complexity_enum)
        required_agents = await self._determine_required_agents(task_type, complexity_enum, input_text, requires_advanced_reasoning, requires_external_knowledge)
        dependencies = await self._identify_dependencies(input_text, context)
        risk_level = await self._assess_risk_level(input_text, task_type, complexity_enum)
        confidence = await self._calculate_confidence(input_text, task_type)
        metadata = await self._extract_metadata(input_text, context)
        metadata['complexity_score_numeric'] = complexity_score # Store numeric score too

        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity_enum,
            estimated_duration=duration,
            required_agents=required_agents,
            dependencies=dependencies,
            risk_level=risk_level,
            confidence=confidence,
            metadata=metadata,
            requires_advanced_reasoning=requires_advanced_reasoning,
            requires_external_knowledge=requires_external_knowledge,
            requires_planning=requires_planning
        )

    async def _detect_task_type(self, input_text: str) -> TaskType:
        text_lower = input_text.lower()
        type_scores: Dict[TaskType, int] = {ttype: 0 for ttype in TaskType}

        # Prioritize more specific task types
        priority_order = [
            TaskType.COMPLEX_REASONING, TaskType.KNOWLEDGE_LOOKUP, TaskType.PLANNING,
            TaskType.BUG_FIX, TaskType.RESEARCH_INTEGRATION, TaskType.SECURITY_AUDIT,
            TaskType.PERFORMANCE_OPTIMIZATION, TaskType.REFACTORING, TaskType.ARCHITECTURE,
            TaskType.TESTING, TaskType.DOCUMENTATION, TaskType.CODE_GENERATION
        ]

        for task_type_check in priority_order:
            patterns = self.task_patterns.get(task_type_check, [])
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return task_type_check # Return first specific match

        # Fallback to general scoring if no specific priority match
        for task_type, patterns in self.task_patterns.items():
            score = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
            type_scores[task_type] = score

        if type_scores:
            # Filter out types with zero scores before max, or handle if all are zero
            positive_scores = {k: v for k,v in type_scores.items() if v > 0}
            if positive_scores:
                return max(positive_scores, key=positive_scores.get)
        return TaskType.UNKNOWN

    async def _analyze_complexity(self, input_text: str, task_type: TaskType,
                                  adv_reasoning: bool, ext_knowledge: bool) -> tuple[TaskComplexity, float]:
        text_lower = input_text.lower()
        base_complexity_map = {
            TaskType.CODE_GENERATION: 0.4, TaskType.BUG_FIX: 0.5, TaskType.REFACTORING: 0.6,
            TaskType.TESTING: 0.3, TaskType.DOCUMENTATION: 0.2, TaskType.ARCHITECTURE: 0.7,
            TaskType.SECURITY_AUDIT: 0.6, TaskType.PERFORMANCE_OPTIMIZATION: 0.6,
            TaskType.RESEARCH_INTEGRATION: 0.8, TaskType.COMPLEX_REASONING: 0.75,
            TaskType.KNOWLEDGE_LOOKUP: 0.3, TaskType.PLANNING: 0.65, TaskType.UNKNOWN: 0.4
        }
        complexity_score = base_complexity_map.get(task_type, 0.5)

        if adv_reasoning: complexity_score = max(complexity_score, 0.7) # Boost complexity if advanced reasoning needed
        if ext_knowledge: complexity_score = max(complexity_score, 0.5) # Boost for external knowledge

        for indicator, weight in self.complexity_indicators.items():
            if indicator in text_lower:
                complexity_score = max(complexity_score, weight)

        complexity_factors = [
            ('algorithm', 0.3), ('machine learning', 0.4), ('ai', 0.3),
            ('database', 0.2), ('api', 0.2), ('microservice', 0.4),
            ('distributed', 0.5), ('concurrent', 0.4), ('async', 0.3),
            ('real-time', 0.4), ('scalable', 0.3), ('enterprise', 0.4)
        ]
        for factor, weight_factor in complexity_factors:
            if factor in text_lower:
                complexity_score += weight_factor * 0.1 # More subtle influence

        complexity_score = min(complexity_score, 1.0)

        # Determine enum based on score, but also consider flags for specific complexity types
        if adv_reasoning and complexity_score >= 0.65: return TaskComplexity.REQUIRES_DEEP_REASONING, complexity_score
        if ext_knowledge and complexity_score >= 0.5: return TaskComplexity.REQUIRES_KNOWLEDGE_GRAPH, complexity_score # Example specific complexity

        if complexity_score < 0.3: return TaskComplexity.SIMPLE, complexity_score
        if complexity_score < 0.6: return TaskComplexity.MODERATE, complexity_score
        if complexity_score < 0.8: return TaskComplexity.COMPLEX, complexity_score
        return TaskComplexity.EXPERT, complexity_score

    async def _estimate_duration(self, input_text: str, task_type: TaskType, complexity: TaskComplexity) -> int:
        # Duration estimation can remain similar, but complexity enum might be more specific now
        base_durations = {
            TaskType.CODE_GENERATION: 120, TaskType.BUG_FIX: 180, TaskType.REFACTORING: 240,
            TaskType.TESTING: 90, TaskType.DOCUMENTATION: 60, TaskType.ARCHITECTURE: 300,
            TaskType.SECURITY_AUDIT: 240, TaskType.PERFORMANCE_OPTIMIZATION: 300,
            TaskType.RESEARCH_INTEGRATION: 360, TaskType.COMPLEX_REASONING: 280,
            TaskType.KNOWLEDGE_LOOKUP: 100, TaskType.PLANNING: 200, TaskType.UNKNOWN: 120
        }
        # Map new specific complexities to multipliers or handle them
        complexity_multipliers = {
            TaskComplexity.SIMPLE: 0.5, TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 2.0, TaskComplexity.EXPERT: 3.0,
            TaskComplexity.REQUIRES_DEEP_REASONING: 2.5, # Example
            TaskComplexity.REQUIRES_KNOWLEDGE_GRAPH: 1.5  # Example
        }
        base_duration = base_durations.get(task_type, 120)
        multiplier = complexity_multipliers.get(complexity, 1.0) # Fallback for unmapped specific complexities
        length_factor = min(len(input_text) / 1000.0, 2.0)
        estimated_duration = int(base_duration * multiplier * (1 + length_factor * 0.5))
        return min(max(estimated_duration, 30), 1800)

    async def _determine_required_agents(self, task_type: TaskType, complexity: TaskComplexity,
                                         input_text: str, adv_reasoning: bool, ext_knowledge: bool) -> List[str]:
        agent_requirements_map = {
            TaskType.CODE_GENERATION: ['codemaster'], TaskType.BUG_FIX: ['codemaster', 'test'],
            TaskType.REFACTORING: ['codemaster', 'refactor'], TaskType.TESTING: ['test'],
            TaskType.DOCUMENTATION: ['document'], TaskType.ARCHITECTURE: ['architect'],
            TaskType.SECURITY_AUDIT: ['security'], TaskType.PERFORMANCE_OPTIMIZATION: ['refactor', 'test'],
            TaskType.RESEARCH_INTEGRATION: ['research', 'codemaster'],
            TaskType.COMPLEX_REASONING: ['codemaster', 'architect'], # Example, could be a 'ReasoningAgent'
            TaskType.KNOWLEDGE_LOOKUP: ['research'], # Example, could be 'KnowledgeAgent'
            TaskType.PLANNING: ['architect', 'codemaster'],
            TaskType.UNKNOWN: ['codemaster']
        }
        required_agents = agent_requirements_map.get(task_type, ['codemaster']).copy()

        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT, TaskComplexity.REQUIRES_DEEP_REASONING]:
            if 'architect' not in required_agents: required_agents.append('architect')
            if 'test' not in required_agents: required_agents.append('test')

        if adv_reasoning and 'ReasoningAgent' not in required_agents: # Assuming a future ReasoningAgent
            # required_agents.append('ReasoningAgent')
            pass # For now, existing agents handle it
        if ext_knowledge and 'research' not in required_agents:
            required_agents.append('research') # ResearchAgent can handle knowledge lookup

        text_lower = input_text.lower()
        keyword_agents_map = {
            'security': ['security'], 'test': ['test'], 'document': ['document'],
            'deploy': ['deploy'], 'performance': ['refactor'],
            'architecture': ['architect'], 'research': ['research']
        }
        for keyword, agents in keyword_agents_map.items():
            if keyword in text_lower:
                for agent in agents:
                    if agent not in required_agents: required_agents.append(agent)
        return list(set(required_agents))

    async def _identify_dependencies(self, input_text: str, context: Dict[str, Any]) -> List[str]:
        dependencies = []
        dependency_patterns = [r'depends on\s+(\w+)', r'requires\s+(\w+)', r'needs\s+(\w+)', r'after\s+(\w+)', r'following\s+(\w+)']
        for pattern in dependency_patterns:
            dependencies.extend(re.findall(pattern, input_text.lower()))
        if context:
            dependencies.extend(context.get('previous_tasks', []))
            dependencies.extend(context.get('required_components', []))
        return list(set(dependencies))

    async def _assess_risk_level(self, input_text: str, task_type: TaskType, complexity: TaskComplexity) -> float:
        # Risk assessment can remain similar, but new task types might have different base risks
        base_risks_map = {
            TaskType.CODE_GENERATION: 0.3, TaskType.BUG_FIX: 0.4, TaskType.REFACTORING: 0.5,
            TaskType.TESTING: 0.2, TaskType.DOCUMENTATION: 0.1, TaskType.ARCHITECTURE: 0.6,
            TaskType.SECURITY_AUDIT: 0.3, TaskType.PERFORMANCE_OPTIMIZATION: 0.5,
            TaskType.RESEARCH_INTEGRATION: 0.7, TaskType.COMPLEX_REASONING: 0.4,
            TaskType.KNOWLEDGE_LOOKUP: 0.2, TaskType.PLANNING: 0.5, TaskType.UNKNOWN: 0.3
        }
        risk_score = base_risks_map.get(task_type, 0.3)
        complexity_risk_map = {
            TaskComplexity.SIMPLE: 0.0, TaskComplexity.MODERATE: 0.1,
            TaskComplexity.COMPLEX: 0.2, TaskComplexity.EXPERT: 0.3,
            TaskComplexity.REQUIRES_DEEP_REASONING: 0.25,
            TaskComplexity.REQUIRES_KNOWLEDGE_GRAPH: 0.15
        }
        risk_score += complexity_risk_map.get(complexity, 0.1)
        high_risk_keywords = ['production', 'live', 'critical', 'important', 'urgent', 'database', 'security', 'authentication', 'payment', 'user data', 'sensitive', 'confidential']
        text_lower = input_text.lower()
        for keyword in high_risk_keywords:
            if keyword in text_lower: risk_score += 0.1
        return min(risk_score, 1.0)

    async def _calculate_confidence(self, input_text: str, task_type: TaskType) -> float:
        # Confidence calculation can remain similar
        confidence = 0.7
        if len(input_text) > 50: confidence += 0.1
        if len(input_text) > 200: confidence += 0.1
        technical_terms = ['function', 'class', 'method', 'algorithm', 'api', 'database', 'framework', 'library', 'module', 'component']
        text_lower = input_text.lower()
        term_count = sum(1 for term in technical_terms if term in text_lower)
        confidence += min(term_count * 0.05, 0.2)
        action_words = ['create', 'build', 'implement', 'fix', 'optimize', 'refactor', 'test', 'document', 'design', 'analyze', 'plan', 'reason', 'lookup']
        action_count = sum(1 for word in action_words if word in text_lower)
        if action_count > 0: confidence += 0.1
        return min(confidence, 1.0)

    async def _extract_metadata(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Metadata extraction can remain similar
        metadata = {
            'input_length': len(input_text),
            'word_count': len(input_text.split()),
            'has_code_snippets': bool(re.search(r'```|`[^`]+`', input_text)),
            'has_urls': bool(re.search(r'https?://', input_text)),
            'has_file_references': bool(re.search(r'\.\w{2,4}(?:\s|$)', input_text)),
            'urgency_indicators': self._detect_urgency(input_text),
            'programming_languages': self._detect_languages(input_text),
            'frameworks_mentioned': self._detect_frameworks(input_text)
        }
        metadata['has_context'] = bool(context)
        if context: metadata['context_keys'] = list(context.keys())
        return metadata

    def _detect_urgency(self, input_text: str) -> List[str]:
        urgency_words = ['urgent', 'asap', 'immediately', 'quickly', 'fast', 'emergency', 'critical', 'important', 'priority']
        return [word for word in urgency_words if word in input_text.lower()]

    def _detect_languages(self, input_text: str) -> List[str]:
        languages = ['python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'typescript', 'php', 'ruby', 'swift', 'kotlin']
        return [lang for lang in languages if lang in input_text.lower()]

    def _detect_frameworks(self, input_text: str) -> List[str]:
        frameworks = ['react', 'vue', 'angular', 'django', 'flask', 'fastapi', 'express', 'spring', 'laravel', 'rails', 'tensorflow', 'pytorch']
        return [framework for framework in frameworks if framework in input_text.lower()]

__all__ = ['TaskAnalyzer', 'TaskAnalysis', 'TaskType', 'TaskComplexity']
