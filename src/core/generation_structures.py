from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

@dataclass
class CodeSpecification:
    '''
    Represents the specification for a piece of code to be generated.
    '''
    spec_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_language: str  # e.g., "python", "javascript"

    # Details to guide the LLM for code generation.
    # This could be a natural language description of the desired logic,
    # function signatures, class structures, input/output examples, etc.
    prompt_details: str

    # Optional summary of the broader context (e.g., project type, existing modules)
    # that might influence how the code should be written.
    context_summary: Optional[str] = None

    # Examples of desired input/output or behavior, if applicable
    examples: Optional[List[Dict[str, Any]]] = field(default_factory=list) # e.g., [{"input": ..., "output": ...}]

    # Specific requirements or constraints (e.g., "must use library X", "avoid recursion")
    constraints: List[str] = field(default_factory=list)


@dataclass
class CodeGenerationResult:
    '''
    Represents the result of a code generation attempt.
    '''
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    specification_id: str # ID of the CodeSpecification this result is for

    generated_code: Optional[str] = None
    language: Optional[str] = None # Should match target_language from spec

    # Optional fields for future enhancements (e.g., by CodeQualityChecker)
    quality_score: Optional[float] = None # e.g., 0.0 to 1.0
    issues_found: List[str] = field(default_factory=list) # e.g., linting errors, style violations
    suggestions: List[str] = field(default_factory=list) # e.g., for improvement

    error_message: Optional[str] = None # If generation failed

    @property
    def succeeded(self) -> bool:
        return self.generated_code is not None and self.error_message is None
