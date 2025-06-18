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
    UNKNOWN = "unknown" # Added for default case

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

@dataclass
class TaskAnalysis:
    task_type: TaskType
    complexity: TaskComplexity
    estimated_duration: int # in seconds
    required_agents: List[str]
    dependencies: List[str]
    risk_level: float # 0.0 to 1.0
    confidence: float # 0.0 to 1.0
    metadata: Dict[str, Any]

class TaskAnalyzer:
    """
    Advanced task analysis system for intelligent routing
    """

    def __init__(self):
        self.task_patterns = self._initialize_patterns()
        self.complexity_indicators = self._initialize_complexity_indicators()

    def _initialize_patterns(self) -> Dict[TaskType, List[str]]:
        return {
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
                r'plan\s+(?:the\s+)?(?:system|architecture)',
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
            ]
        }

    def _initialize_complexity_indicators(self) -> Dict[str, float]:
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
        complexity = await self._analyze_complexity(input_text, task_type)
        duration = await self._estimate_duration(input_text, task_type, complexity)
        required_agents = await self._determine_required_agents(task_type, complexity, input_text)
        dependencies = await self._identify_dependencies(input_text, context)
        risk_level = await self._assess_risk_level(input_text, task_type, complexity)
        confidence = await self._calculate_confidence(input_text, task_type)
        metadata = await self._extract_metadata(input_text, context)

        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity,
            estimated_duration=duration,
            required_agents=required_agents,
            dependencies=dependencies,
            risk_level=risk_level,
            confidence=confidence,
            metadata=metadata
        )

    async def _detect_task_type(self, input_text: str) -> TaskType:
        text_lower = input_text.lower()
        type_scores = {}
        for task_type, patterns in self.task_patterns.items():
            score = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
            type_scores[task_type] = score

        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 0:
                return best_type
        return TaskType.UNKNOWN

    async def _analyze_complexity(self, input_text: str, task_type: TaskType) -> TaskComplexity:
        text_lower = input_text.lower()
        base_complexity_map = {
            TaskType.CODE_GENERATION: 0.4, TaskType.BUG_FIX: 0.5, TaskType.REFACTORING: 0.6,
            TaskType.TESTING: 0.3, TaskType.DOCUMENTATION: 0.2, TaskType.ARCHITECTURE: 0.8,
            TaskType.SECURITY_AUDIT: 0.7, TaskType.PERFORMANCE_OPTIMIZATION: 0.7,
            TaskType.RESEARCH_INTEGRATION: 0.9, TaskType.UNKNOWN: 0.4
        }
        complexity_score = base_complexity_map.get(task_type, 0.5)

        for indicator, weight in self.complexity_indicators.items():
            if indicator in text_lower:
                complexity_score = max(complexity_score, weight)

        complexity_factors = [
            ('algorithm', 0.3), ('machine learning', 0.4), ('ai', 0.3),
            ('database', 0.2), ('api', 0.2), ('microservice', 0.4),
            ('distributed', 0.5), ('concurrent', 0.4), ('async', 0.3),
            ('real-time', 0.4), ('scalable', 0.3), ('enterprise', 0.4)
        ]
        for factor, weight in complexity_factors:
            if factor in text_lower:
                complexity_score += weight * 0.1

        complexity_score = min(complexity_score, 1.0)

        if complexity_score < 0.3: return TaskComplexity.SIMPLE
        if complexity_score < 0.6: return TaskComplexity.MODERATE
        if complexity_score < 0.8: return TaskComplexity.COMPLEX
        return TaskComplexity.EXPERT

    async def _estimate_duration(self, input_text: str, task_type: TaskType, complexity: TaskComplexity) -> int:
        base_durations = {
            TaskType.CODE_GENERATION: 120, TaskType.BUG_FIX: 180, TaskType.REFACTORING: 240,
            TaskType.TESTING: 90, TaskType.DOCUMENTATION: 60, TaskType.ARCHITECTURE: 300,
            TaskType.SECURITY_AUDIT: 240, TaskType.PERFORMANCE_OPTIMIZATION: 300,
            TaskType.RESEARCH_INTEGRATION: 360, TaskType.UNKNOWN: 120
        }
        complexity_multipliers = {
            TaskComplexity.SIMPLE: 0.5, TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 2.0, TaskComplexity.EXPERT: 3.0
        }
        base_duration = base_durations.get(task_type, 120)
        multiplier = complexity_multipliers.get(complexity, 1.0)
        length_factor = min(len(input_text) / 1000.0, 2.0) # ensure float division
        estimated_duration = int(base_duration * multiplier * (1 + length_factor * 0.5))
        return min(max(estimated_duration, 30), 1800)

    async def _determine_required_agents(self, task_type: TaskType, complexity: TaskComplexity, input_text: str) -> List[str]:
        agent_requirements_map = {
            TaskType.CODE_GENERATION: ['codemaster'], TaskType.BUG_FIX: ['codemaster', 'test'],
            TaskType.REFACTORING: ['codemaster', 'refactor'], TaskType.TESTING: ['test'],
            TaskType.DOCUMENTATION: ['document'], TaskType.ARCHITECTURE: ['architect'],
            TaskType.SECURITY_AUDIT: ['security'], TaskType.PERFORMANCE_OPTIMIZATION: ['refactor', 'test'],
            TaskType.RESEARCH_INTEGRATION: ['research', 'codemaster'], TaskType.UNKNOWN: ['codemaster']
        }
        required_agents = agent_requirements_map.get(task_type, ['codemaster']).copy()

        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]:
            if 'architect' not in required_agents: required_agents.append('architect')
            if 'test' not in required_agents: required_agents.append('test')

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
        base_risks_map = {
            TaskType.CODE_GENERATION: 0.3, TaskType.BUG_FIX: 0.4, TaskType.REFACTORING: 0.5,
            TaskType.TESTING: 0.2, TaskType.DOCUMENTATION: 0.1, TaskType.ARCHITECTURE: 0.6,
            TaskType.SECURITY_AUDIT: 0.3, TaskType.PERFORMANCE_OPTIMIZATION: 0.5,
            TaskType.RESEARCH_INTEGRATION: 0.7, TaskType.UNKNOWN: 0.3
        }
        risk_score = base_risks_map.get(task_type, 0.3)
        complexity_risk_map = {
            TaskComplexity.SIMPLE: 0.0, TaskComplexity.MODERATE: 0.1,
            TaskComplexity.COMPLEX: 0.2, TaskComplexity.EXPERT: 0.3
        }
        risk_score += complexity_risk_map.get(complexity, 0.1)
        high_risk_keywords = ['production', 'live', 'critical', 'important', 'urgent', 'database', 'security', 'authentication', 'payment', 'user data', 'sensitive', 'confidential']
        text_lower = input_text.lower()
        for keyword in high_risk_keywords:
            if keyword in text_lower: risk_score += 0.1
        return min(risk_score, 1.0)

    async def _calculate_confidence(self, input_text: str, task_type: TaskType) -> float:
        confidence = 0.7
        if len(input_text) > 50: confidence += 0.1
        if len(input_text) > 200: confidence += 0.1
        technical_terms = ['function', 'class', 'method', 'algorithm', 'api', 'database', 'framework', 'library', 'module', 'component']
        text_lower = input_text.lower()
        term_count = sum(1 for term in technical_terms if term in text_lower)
        confidence += min(term_count * 0.05, 0.2)
        action_words = ['create', 'build', 'implement', 'fix', 'optimize', 'refactor', 'test', 'document', 'design', 'analyze']
        action_count = sum(1 for word in action_words if word in text_lower)
        if action_count > 0: confidence += 0.1
        return min(confidence, 1.0)

    async def _extract_metadata(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {
            'input_length': len(input_text),
            'word_count': len(input_text.split()),
            'has_code_snippets': bool(re.search(r'```|`[^`]+`', input_text)),
            'has_urls': bool(re.search(r'https?://', input_text)),
            'has_file_references': bool(re.search(r'\.\w{2,4}(?:\s|$)', input_text)), # Corrected regex escape
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
