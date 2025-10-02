"""
Structured audit logging system with tamper detection.

Based on research:
- Schneier & Kelsey, 1999 - Secure Audit Logs
- NIST SP 800-92 - Guide to Computer Security Log Management
- Zawoad & Hasan, 2015, arXiv:1503.08052 - Cloud Forensics
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib
import json
from typing import Any, Dict, Optional
import structlog


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication & Authorization
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    API_KEY_CREATED = "api_key.created"
    API_KEY_ROTATED = "api_key.rotated"
    API_KEY_REVOKED = "api_key.revoked"
    
    # Provider Operations
    PROVIDER_SELECTED = "provider.selected"
    PROVIDER_REQUEST = "provider.request"
    PROVIDER_RESPONSE = "provider.response"
    PROVIDER_ERROR = "provider.error"
    PROVIDER_FALLBACK = "provider.fallback"
    
    # Model Operations
    MODEL_SELECTED = "model.selected"
    ROUTING_DECISION = "routing.decision"
    ENSEMBLE_INVOKED = "ensemble.invoked"
    
    # Configuration Changes
    CONFIG_UPDATED = "config.updated"
    RULE_ADDED = "rule.added"
    RULE_MODIFIED = "rule.modified"
    RULE_DELETED = "rule.deleted"
    
    # Rate Limiting
    RATE_LIMIT_HIT = "rate_limit.hit"
    RATE_LIMIT_RESET = "rate_limit.reset"
    
    # Security
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    ACCESS_DENIED = "security.denied"


@dataclass
class AuditEvent:
    """Structured audit event with tamper detection."""
    timestamp: str  # ISO 8601
    event_type: AuditEventType
    actor: str  # User ID or system component
    action: str  # Human-readable action description
    resource_type: str  # e.g., "provider", "model", "api_key"
    resource_id: str  # e.g., "openai", "gpt-4", "key-123"
    status: str  # "success", "failure", "error"
    metadata: Dict[str, Any]
    request_id: str  # Correlation ID for distributed tracing
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Security fields for tamper detection
    previous_hash: Optional[str] = None  # Hash of previous log entry
    entry_hash: Optional[str] = None  # Hash of this entry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        return data
    
    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of entry for tamper detection.
        
        Creates a canonical representation and hashes it to detect
        any modifications to the audit log.
        """
        # Create canonical representation (sorted keys for consistency)
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


class AuditLogger:
    """
    Centralized audit logging with tamper detection and structured output.
    
    Features:
    - Structured JSON logging via structlog
    - Hash chaining for tamper detection (Schneier & Kelsey pattern)
    - Support for correlation IDs (distributed tracing)
    - Comprehensive event types for security compliance
    
    Based on:
    - Schneier & Kelsey, 1999 - Secure audit log design
    - NIST SP 800-92 - Computer Security Log Management
    """
    
    def __init__(self, output_path: str = "audit.log", 
                 enable_chaining: bool = True):
        """
        Initialize audit logger.
        
        Args:
            output_path: Path to audit log file
            enable_chaining: Enable hash chaining for tamper detection
        """
        self.logger = structlog.get_logger("audit")
        self.output_path = output_path
        self.enable_chaining = enable_chaining
        self.last_hash: Optional[str] = None
        
        # Configure structured output
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ]
        )
    
    def log_event(self, event: AuditEvent) -> None:
        """
        Log an audit event with optional chaining.
        
        Args:
            event: The audit event to log
        """
        # Add timestamp if not set
        if not event.timestamp:
            event.timestamp = datetime.utcnow().isoformat()
        
        # Add hash chaining for tamper detection
        if self.enable_chaining:
            event.previous_hash = self.last_hash
            event.entry_hash = event.compute_hash()
            self.last_hash = event.entry_hash
        
        # Log with structured logger
        self.logger.info(
            "audit_event",
            **event.to_dict()
        )
    
    def log_provider_request(self, provider: str, model: str, 
                           request_id: str, actor: str,
                           metadata: Optional[Dict] = None) -> None:
        """
        Log a provider API request.
        
        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model identifier
            request_id: Correlation ID for request tracking
            actor: User or system making the request
            metadata: Additional context
        """
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=AuditEventType.PROVIDER_REQUEST,
            actor=actor,
            action=f"Request to {provider}",
            resource_type="provider",
            resource_id=provider,
            status="initiated",
            metadata=metadata or {"model": model},
            request_id=request_id
        )
        self.log_event(event)
    
    def log_routing_decision(self, selected_provider: str, 
                           routing_rule: str, request_id: str,
                           alternatives: list, metadata: Optional[Dict] = None) -> None:
        """
        Log a routing decision made by the router.
        
        Args:
            selected_provider: Provider that was selected
            routing_rule: Rule that triggered the selection
            request_id: Correlation ID
            alternatives: List of alternative providers considered
            metadata: Additional routing context
        """
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=AuditEventType.ROUTING_DECISION,
            actor="system.router",
            action=f"Selected {selected_provider} via rule {routing_rule}",
            resource_type="routing",
            resource_id=routing_rule,
            status="success",
            metadata={
                "selected": selected_provider,
                "alternatives": alternatives,
                **(metadata or {})
            },
            request_id=request_id
        )
        self.log_event(event)
    
    def log_auth_attempt(self, user_id: str, success: bool,
                       ip_address: str, reason: Optional[str] = None) -> None:
        """
        Log an authentication attempt.
        
        Args:
            user_id: User identifier
            success: Whether authentication succeeded
            ip_address: Source IP address
            reason: Failure reason if applicable
        """
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=AuditEventType.AUTH_SUCCESS if success else AuditEventType.AUTH_FAILURE,
            actor=user_id,
            action="Authentication attempt",
            resource_type="auth",
            resource_id=user_id,
            status="success" if success else "failure",
            metadata={"reason": reason} if reason else {},
            request_id=f"auth-{datetime.utcnow().timestamp()}",
            ip_address=ip_address
        )
        self.log_event(event)
    
    def log_rate_limit(self, user_id: str, endpoint: str,
                      limit: int, current: int, request_id: str) -> None:
        """
        Log a rate limit event.
        
        Args:
            user_id: User who hit the rate limit
            endpoint: API endpoint affected
            limit: Rate limit threshold
            current: Current request count
            request_id: Correlation ID
        """
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=AuditEventType.RATE_LIMIT_HIT,
            actor=user_id,
            action=f"Rate limit hit on {endpoint}",
            resource_type="rate_limit",
            resource_id=endpoint,
            status="blocked",
            metadata={
                "limit": limit,
                "current": current,
                "exceeded_by": current - limit
            },
            request_id=request_id
        )
        self.log_event(event)
    
    def log_config_change(self, actor: str, config_key: str,
                        old_value: Any, new_value: Any,
                        request_id: str) -> None:
        """
        Log a configuration change.
        
        Args:
            actor: User or system making the change
            config_key: Configuration key being modified
            old_value: Previous value
            new_value: New value
            request_id: Correlation ID
        """
        # Redact sensitive values
        def redact_if_sensitive(key: str, value: Any) -> str:
            sensitive_keys = ['password', 'secret', 'key', 'token']
            if any(s in key.lower() for s in sensitive_keys):
                return "[REDACTED]"
            return str(value)
        
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=AuditEventType.CONFIG_UPDATED,
            actor=actor,
            action=f"Updated configuration: {config_key}",
            resource_type="config",
            resource_id=config_key,
            status="success",
            metadata={
                "old_value": redact_if_sensitive(config_key, old_value),
                "new_value": redact_if_sensitive(config_key, new_value)
            },
            request_id=request_id
        )
        self.log_event(event)
    
    def log_provider_error(self, provider: str, error_type: str,
                         error_message: str, request_id: str) -> None:
        """
        Log a provider error.
        
        Args:
            provider: Provider that encountered the error
            error_type: Type of error (e.g., "rate_limit", "auth_failure")
            error_message: Error description
            request_id: Correlation ID
        """
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=AuditEventType.PROVIDER_ERROR,
            actor="system",
            action=f"Provider error: {provider}",
            resource_type="provider",
            resource_id=provider,
            status="error",
            metadata={
                "error_type": error_type,
                "error_message": error_message
            },
            request_id=request_id
        )
        self.log_event(event)
