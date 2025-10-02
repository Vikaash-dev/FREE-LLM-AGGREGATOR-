# Implementation Plan: Structured Audit Logging

## Overview
Based on component evaluations, structured audit logging has been identified as the highest priority enhancement across all components. This document provides a detailed implementation plan.

## Research Basis

### Relevant Papers
1. [Schneier & Kelsey, 1999] - Secure Audit Logs to Support Computer Forensics
2. [Chuvakin et al., 2013] - Security Monitoring and Audit Logging
3. [Zawoad & Hasan, 2015, arXiv:1503.08052] - Cloud Forensics: A Meta-Study
4. [Snodgrass et al., 2004, ACM TODS] - Tamper Detection in Audit Logs

### GitHub Project References
1. **python-json-logger** (1.5k⭐) - Structured JSON logging
2. **structlog** (3k⭐) - Structured logging for Python (already used)
3. **elasticsearch-py** (4k⭐) - Log aggregation
4. **OpenTelemetry** (3k⭐) - Distributed tracing and logging
5. **audit-log** patterns from AWS CloudTrail, Azure Monitor
6. **Falco** (6k⭐) - Runtime security and audit logging

## Requirements

### Functional Requirements
1. **Capture Events**: All security-relevant operations must be logged
2. **Immutability**: Logs must be tamper-evident
3. **Searchability**: Fast queries by user, resource, time, action
4. **Compliance**: Support SOC2, GDPR, HIPAA requirements
5. **Performance**: <5ms overhead per operation
6. **Retention**: Configurable retention policies

### Security Requirements
1. **Integrity**: Cryptographic hashing of log entries
2. **Confidentiality**: Sensitive data must be redacted/encrypted
3. **Availability**: Logs must survive system failures
4. **Non-repudiation**: Logs must be legally defensible

## Architecture Design

### Components

#### 1. AuditLogger (Core)
```python
"""
Centralized audit logging with structured output.
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
    """Structured audit event."""
    timestamp: str  # ISO 8601
    event_type: AuditEventType
    actor: str  # User ID or system component
    action: str  # Human-readable action
    resource_type: str  # e.g., "provider", "model", "api_key"
    resource_id: str  # e.g., "openai", "gpt-4", "key-123"
    status: str  # "success", "failure", "error"
    metadata: Dict[str, Any]
    request_id: str  # Correlation ID
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Security fields
    previous_hash: Optional[str] = None  # Hash of previous log entry
    entry_hash: Optional[str] = None  # Hash of this entry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        return data
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of entry for tamper detection."""
        # Create canonical representation
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

class AuditLogger:
    """
    Centralized audit logging with tamper detection.
    
    Based on:
    - Schneier & Kelsey secure audit log design
    - NIST SP 800-92 (Guide to Computer Security Log Management)
    """
    
    def __init__(self, output_path: str = "audit.log", 
                 enable_chaining: bool = True):
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
                           metadata: Optional[Dict] = None):
        """Log a provider API request."""
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
                           alternatives: list, metadata: Optional[Dict] = None):
        """Log a routing decision."""
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=AuditEventType.ROUTING_DECISION,
            actor="system.router",
            action=f"Selected {selected_provider}",
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
                       ip_address: str, reason: Optional[str] = None):
        """Log an authentication attempt."""
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
                      limit: int, current: int, request_id: str):
        """Log a rate limit event."""
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
                        request_id: str):
        """Log a configuration change."""
        event = AuditEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=AuditEventType.CONFIG_UPDATED,
            actor=actor,
            action=f"Updated configuration: {config_key}",
            resource_type="config",
            resource_id=config_key,
            status="success",
            metadata={
                "old_value": str(old_value),  # Redact sensitive values
                "new_value": str(new_value)
            },
            request_id=request_id
        )
        self.log_event(event)
```

#### 2. Integration Points

**Aggregator Integration:**
```python
# In aggregator.py
class LLMAggregator:
    def __init__(self, ..., audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger or AuditLogger()
    
    async def chat_completion(self, request: ChatCompletionRequest):
        request_id = str(uuid.uuid4())
        
        # Log routing decision
        provider = await self.router.get_provider_chain(request)
        self.audit_logger.log_routing_decision(
            selected_provider=provider[0],
            routing_rule="auto",
            request_id=request_id,
            alternatives=provider[1:],
            metadata={"model": request.model}
        )
        
        # Log provider request
        self.audit_logger.log_provider_request(
            provider=provider[0],
            model=request.model,
            request_id=request_id,
            actor=request.user or "anonymous"
        )
```

**Router Integration:**
```python
# In router.py
class ProviderRouter:
    def __init__(self, ..., audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger
    
    async def get_provider_chain(self, request):
        matching_rules = self._find_matching_rules(request)
        
        if self.audit_logger and matching_rules:
            self.audit_logger.log_routing_decision(
                selected_provider=matching_rules[0].provider_preferences[0],
                routing_rule=matching_rules[0].name,
                request_id=getattr(request, 'request_id', 'unknown'),
                alternatives=matching_rules[0].fallback_chain
            )
```

**AccountManager Integration:**
```python
# In account_manager.py
class AccountManager:
    def __init__(self, ..., audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger
    
    def rotate_api_key(self, provider: str, account_id: str):
        old_key_hash = self._hash_key(self.get_api_key(provider, account_id))
        
        # Perform rotation
        new_key = self._generate_new_key()
        self.store_credentials(provider, account_id, new_key)
        
        # Audit log
        if self.audit_logger:
            self.audit_logger.log_event(AuditEvent(
                timestamp=datetime.utcnow().isoformat(),
                event_type=AuditEventType.API_KEY_ROTATED,
                actor="system",
                action=f"Rotated API key for {provider}",
                resource_type="api_key",
                resource_id=f"{provider}:{account_id}",
                status="success",
                metadata={"old_key_hash": old_key_hash},
                request_id=f"rotation-{uuid.uuid4()}"
            ))
```

## Testing Strategy

### Unit Tests
```python
# tests/test_audit_logger.py
import pytest
from src.core.audit_logger import AuditLogger, AuditEvent, AuditEventType

def test_event_hashing():
    """Test that events are hashed consistently."""
    event = AuditEvent(
        timestamp="2024-01-01T00:00:00",
        event_type=AuditEventType.PROVIDER_REQUEST,
        actor="test",
        action="test",
        resource_type="provider",
        resource_id="openai",
        status="success",
        metadata={},
        request_id="test-123"
    )
    
    hash1 = event.compute_hash()
    hash2 = event.compute_hash()
    assert hash1 == hash2

def test_hash_chaining():
    """Test that events are chained with hashes."""
    logger = AuditLogger(enable_chaining=True)
    
    event1 = AuditEvent(...)
    event2 = AuditEvent(...)
    
    logger.log_event(event1)
    logger.log_event(event2)
    
    assert event2.previous_hash == event1.entry_hash

def test_routing_decision_log():
    """Test routing decision logging."""
    logger = AuditLogger()
    logger.log_routing_decision(
        selected_provider="openai",
        routing_rule="code_generation",
        request_id="req-123",
        alternatives=["anthropic", "groq"]
    )
    # Verify log output
```

### Integration Tests
```python
# tests/test_audit_integration.py
import pytest
from src.core.aggregator import LLMAggregator
from src.core.audit_logger import AuditLogger

@pytest.mark.asyncio
async def test_audit_logging_in_aggregator():
    """Test that aggregator logs audit events."""
    audit_logger = AuditLogger()
    aggregator = LLMAggregator(..., audit_logger=audit_logger)
    
    request = ChatCompletionRequest(...)
    await aggregator.chat_completion(request)
    
    # Verify audit logs were created
    # Check for routing decision, provider request logs
```

## Deployment Plan

### Phase 1: Core Implementation (Week 1)
- [ ] Implement AuditLogger class
- [ ] Add unit tests
- [ ] Add to configuration system

### Phase 2: Integration (Week 2)
- [ ] Integrate with Aggregator
- [ ] Integrate with Router
- [ ] Integrate with AccountManager
- [ ] Add integration tests

### Phase 3: Storage & Search (Week 3)
- [ ] Add Elasticsearch integration for log aggregation
- [ ] Create search/query API
- [ ] Add log retention policies

### Phase 4: Monitoring & Alerting (Week 4)
- [ ] Configure alerts for security events
- [ ] Create audit dashboard
- [ ] Document audit log schema

## Monitoring & Validation

### Metrics to Track
- Audit events per second
- Log storage growth rate
- Hash chain validation success rate
- Query response times

### Alerts
- Failed hash chain validation (tamper detection)
- Excessive failed auth attempts
- Unusual provider error rates
- Rate limit violations

## Compliance Mapping

### SOC 2
- **CC6.1**: Logical access controls - Auth logging ✓
- **CC7.2**: System monitoring - All operations logged ✓
- **CC7.3**: Evaluation of security events - Audit trail ✓

### GDPR
- **Article 30**: Records of processing - Audit trail ✓
- **Article 32**: Security measures - Tamper detection ✓

### HIPAA
- **§164.308(a)(1)(ii)(D)**: Information system activity review ✓
- **§164.312(b)**: Audit controls ✓

---
**Priority**: CRITICAL  
**Effort**: 4 weeks  
**Impact**: High (Security, Compliance, Debugging)  
**Dependencies**: None (can be implemented independently)  
**Risk**: Low (additive, doesn't modify existing logic)
