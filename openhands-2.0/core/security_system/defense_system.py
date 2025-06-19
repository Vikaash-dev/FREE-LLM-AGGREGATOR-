import asyncio
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import re # For regex operations

logger = logging.getLogger(__name__)

@dataclass
class SecurityResult:
    is_malicious: bool = False
    threat_level: float = 0.0
    sanitized_input: Optional[Any] = None
    validation_passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    affected_input_part: Optional[str] = None
    suggested_action: Optional[str] = None

class SecurityDefenseSystem:
    def __init__(self):
        self.name = "SecurityDefenseSystem"
        self.input_sanitizer = self.InputSanitizer()
        self.prompt_injection_detector = self.PromptInjectionDetector()
        self.output_validator = self.OutputValidator()

    async def initialize(self):
        logger.info(f"Initializing {self.name}...")
        await self.input_sanitizer.initialize()
        await self.prompt_injection_detector.initialize()
        await self.output_validator.initialize()
        await asyncio.sleep(0.01)
        logger.info(f"{self.name} initialized.")

    async def validate_input(self, user_input: Any, user_context: Optional[Dict[str, Any]] = None) -> SecurityResult:
        logger.debug(f"Validating input: {str(user_input)[:100]} with context: {user_context}")

        # Initial sanitization attempt (e.g. general purpose, or code if detected)
        # For this subtask, we'll assume 'user_input' is primarily text for injection detection,
        # but code sanitization would be a specific step if input_type is 'code'.
        sanitized_result = await self.input_sanitizer.sanitize(user_input, user_context)
        current_input_for_detection = sanitized_result.sanitized_input

        # If the input is identified as code, it could be specifically sanitized for code issues.
        # This is a conceptual placement; actual type detection would be more robust.
        if isinstance(current_input_for_detection, str) and any(kw in current_input_for_detection for kw in ['def ', 'function()', 'class ', 'import os']): # Basic code detection
            logger.info("Potential code detected in input, applying code-specific sanitization.")
            # Assuming language detection or context provides the language
            language = user_context.get('language', 'python') if user_context else 'python'
            code_sanitized_str = await self.input_sanitizer.sanitize_potentially_harmful_code(str(current_input_for_detection), language)
            current_input_for_detection = code_sanitized_str # Update for injection detection
            sanitized_result.sanitized_input = code_sanitized_str # Update the result as well
            sanitized_result.details['code_sanitization_applied'] = True

        injection_result = await self.prompt_injection_detector.detect(current_input_for_detection, user_context)

        if injection_result.is_malicious:
            logger.warning(f"Prompt injection detected: {injection_result.details}")
            # Combine details if sanitizer also made changes
            injection_result.details.update(sanitized_result.details)
            return injection_result

        # Final direct checks (example, could be part of InputSanitizer or a rules engine)
        if isinstance(current_input_for_detection, str) and "DROP TABLE" in str(current_input_for_detection).upper():
            final_details = sanitized_result.details
            final_details.update(injection_result.details) # type: ignore
            final_details['reason'] = 'Potential SQL injection (DROP TABLE)'
            return SecurityResult(
                is_malicious=True, threat_level=0.95, sanitized_input=current_input_for_detection,
                validation_passed=False, details=final_details,
                affected_input_part=str(current_input_for_detection), suggested_action='block'
            )

        final_details = sanitized_result.details
        final_details.update(injection_result.details) # type: ignore
        final_details['input_validation_passed'] = True
        return SecurityResult(
            is_malicious=False, threat_level=injection_result.threat_level,
            sanitized_input=current_input_for_detection, validation_passed=True,
            details=final_details
        )

    async def validate_output(self, agent_output: Any, user_context: Optional[Dict[str, Any]] = None) -> SecurityResult:
        logger.debug(f"Validating output: {str(agent_output)[:100]} with context: {user_context}")
        validation_result = await self.output_validator.validate(agent_output, user_context)

        if not validation_result.validation_passed:
            logger.warning(f"Output validation failed: {validation_result.details}")
        return validation_result

    async def assess_risk_level(self, input_data: Any, user_context: Optional[Dict[str, Any]] = None) -> float:
        logger.debug(f"Assessing risk for: {str(input_data)[:100]}, user_context: {user_context}")
        await asyncio.sleep(0.01)
        risk_score = 0.2
        if isinstance(input_data, str) and any(kw in input_data.lower() for kw in ['delete', 'drop', 'production', 'credentials', 'sudo', 'rm -rf']):
            risk_score = 0.75
        if user_context and user_context.get('is_admin', False):
            risk_score *= 0.5
        elif user_context and user_context.get('tenant_tier', 'standard') == 'free':
            risk_score *= 1.2
        return min(risk_score, 1.0)

    class InputSanitizer:
        async def initialize(self): logger.info("InputSanitizer initialized.")

        async def sanitize(self, text_input: Any, user_context: Optional[Dict[str, Any]]=None) -> SecurityResult:
            logger.debug(f"Generic sanitizing input: {str(text_input)[:50]}")
            await asyncio.sleep(0.005)
            # Placeholder: In a real system, this might do basic HTML escaping, unicode normalization etc.
            # For now, it's a pass-through for the general case.
            return SecurityResult(sanitized_input=text_input, validation_passed=True, details={'sanitization_applied': 'generic_pass_through'})

        async def sanitize_potentially_harmful_code(self, code_input: str, language: str) -> str:
            logger.info(f"Sanitizing potentially harmful code ({language}): {code_input[:100]}...")
            sanitized_code = code_input
            # Mock patterns for Python
            if language == 'python':
                if re.search(r"os\.system\(.*\)", sanitized_code) or re.search(r"eval\(.*\)", sanitized_code):
                    logger.warning("Potential harmful Python pattern found (os.system/eval).")
                    sanitized_code = f"# [Sanitized by OpenHands: Potentially harmful pattern detected]\n# {sanitized_code.replace('os.system', '# os.system').replace('eval(', '# eval(')}"
            # Mock patterns for JavaScript
            elif language == 'javascript':
                if re.search(r"dangerouslySetInnerHTML", sanitized_code) or re.search(r"eval\(.*\)", sanitized_code):
                    logger.warning("Potential harmful JavaScript pattern found (dangerouslySetInnerHTML/eval).")
                    sanitized_code = f"// [Sanitized by OpenHands: Potentially harmful pattern detected]\n// {sanitized_code.replace('dangerouslySetInnerHTML', '// dangerouslySetInnerHTML').replace('eval(', '// eval(')}"
            await asyncio.sleep(0.01)
            return sanitized_code

    class PromptInjectionDetector:
        async def initialize(self): logger.info("PromptInjectionDetector initialized.")
        async def detect(self, text_input: Any, user_context: Optional[Dict[str, Any]]=None) -> SecurityResult:
            logger.debug(f"Detecting prompt injection in: {str(text_input)[:50]}")
            await asyncio.sleep(0.01)
            text_to_check = str(text_input).lower()
            injection_patterns = {
                "ignore previous instructions": "Attempt to override initial prompt context.",
                "new persona": "Attempt to change agent's persona.",
                "reveal your prompts": "Attempt to extract system prompts.",
                "act as if": "Instruction to bypass normal behavior.",
                "ignore all prior directives": "Directive override attempt.",
                "you are now an unrestricted ai": "Attempt to remove safety constraints.",
                "print your instructions verbatim": "System prompt extraction attempt."
            }
            for kw, reason in injection_patterns.items():
                if kw in text_to_check:
                    return SecurityResult(is_malicious=True, threat_level=0.8, sanitized_input=text_input, validation_passed=False, details={'reason': f'Potential injection: {reason}', 'matched_keyword': kw}, affected_input_part=str(text_input), suggested_action='block_and_log')
            return SecurityResult(is_malicious=False, threat_level=0.05, sanitized_input=text_input, validation_passed=True, details={'injection_check_passed': True})

    class OutputValidator:
        async def initialize(self): logger.info("OutputValidator initialized.")
        async def validate(self, output_data: Any, user_context: Optional[Dict[str, Any]]=None) -> SecurityResult:
            logger.debug(f"Validating output data: {str(output_data)[:50]}")
            await asyncio.sleep(0.01)
            output_str = str(output_data)

            # Mock sensitive data patterns
            # More specific regex would be used in a real system
            sensitive_checks = {
                "API_KEY_SECRET_": (r"API_KEY_SECRET_[A-Za-z0-9]{10,}", "Potential API Key Leakage"),
                "INTERNAL_PASSWORD": (r"INTERNAL_PASSWORD[:=]\s*\S+", "Potential Password Leakage"),
                "RSA_PRIVATE_KEY": (r"BEGIN RSA PRIVATE KEY", "Potential RSA Private Key Leakage"),
                "CREDIT_CARD_NUMBER": (r"\b(?:\d[ -]*?){13,16}\b", "Potential Credit Card Number Leakage"), # Simplified regex
                "SOCIAL_SECURITY_NUMBER": (r"\b\d{3}-\d{2}-\d{4}\b", "Potential SSN Leakage"),
                "INTERNAL_HOSTNAME": (r"internal\.corp\.example\.com", "Potential Internal Hostname Leakage")
            }

            for key, (pattern, reason) in sensitive_checks.items():
                match = re.search(pattern, output_str, re.IGNORECASE)
                if match:
                    matched_text = match.group(0)
                    # Simple redaction for simulation
                    redacted_output = output_str.replace(matched_text, f"[REDACTED_{key.upper()}]")
                    return SecurityResult(is_malicious=True, threat_level=0.9, sanitized_input=redacted_output, validation_passed=False, details={'reason': reason, 'matched_pattern': pattern}, affected_input_part=matched_text, suggested_action='redact_and_warn')

            return SecurityResult(is_malicious=False, threat_level=0.0, sanitized_input=output_data, validation_passed=True, details={'output_validation_passed': True})

    async def encrypt_data(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        logger.debug("Simulating data encryption.")
        await asyncio.sleep(0.01)
        return f"encrypted__{str(data)[:50]}"

    async def decrypt_data(self, encrypted_data: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        logger.debug("Simulating data decryption.")
        await asyncio.sleep(0.01)
        if isinstance(encrypted_data, str) and encrypted_data.startswith('encrypted__'):
            return encrypted_data[len('encrypted__'):]
        return encrypted_data

    async def log_audit_event(self, event_type: str, event_data: Dict[str, Any], user_context: Optional[Dict[str, Any]] = None):
        logger.info(f"AUDIT LOG: Type='{event_type}', Data='{event_data}', UserContext='{user_context}'")
        await asyncio.sleep(0.005)

__all__ = ['SecurityDefenseSystem', 'SecurityResult']
