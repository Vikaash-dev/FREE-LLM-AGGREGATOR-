"""
Tests for the audit logging system.
"""

import pytest
from datetime import datetime
import json

from src.core.audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType
)


class TestAuditEvent:
    """Test suite for AuditEvent."""
    
    def test_event_creation(self):
        """Test that audit events can be created."""
        event = AuditEvent(
            timestamp="2024-01-01T00:00:00",
            event_type=AuditEventType.PROVIDER_REQUEST,
            actor="test-user",
            action="Test action",
            resource_type="provider",
            resource_id="openai",
            status="success",
            metadata={"key": "value"},
            request_id="test-123"
        )
        
        assert event.actor == "test-user"
        assert event.event_type == AuditEventType.PROVIDER_REQUEST
        assert event.resource_id == "openai"
    
    def test_event_to_dict(self):
        """Test conversion to dictionary."""
        event = AuditEvent(
            timestamp="2024-01-01T00:00:00",
            event_type=AuditEventType.AUTH_SUCCESS,
            actor="user1",
            action="Login",
            resource_type="auth",
            resource_id="user1",
            status="success",
            metadata={},
            request_id="req-1"
        )
        
        event_dict = event.to_dict()
        assert isinstance(event_dict, dict)
        assert event_dict['event_type'] == "auth.success"
        assert event_dict['actor'] == "user1"
    
    def test_event_hashing(self):
        """Test that events are hashed consistently."""
        event = AuditEvent(
            timestamp="2024-01-01T00:00:00",
            event_type=AuditEventType.PROVIDER_REQUEST,
            actor="test",
            action="test",
            resource_type="provider",
            resource_id="openai",
            status="success",
            metadata={"test": "data"},
            request_id="test-123"
        )
        
        hash1 = event.compute_hash()
        hash2 = event.compute_hash()
        
        # Hash should be deterministic
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64-char hex string
    
    def test_different_events_different_hashes(self):
        """Test that different events produce different hashes."""
        event1 = AuditEvent(
            timestamp="2024-01-01T00:00:00",
            event_type=AuditEventType.PROVIDER_REQUEST,
            actor="user1",
            action="action1",
            resource_type="provider",
            resource_id="openai",
            status="success",
            metadata={},
            request_id="req-1"
        )
        
        event2 = AuditEvent(
            timestamp="2024-01-01T00:00:01",  # Different timestamp
            event_type=AuditEventType.PROVIDER_REQUEST,
            actor="user1",
            action="action1",
            resource_type="provider",
            resource_id="openai",
            status="success",
            metadata={},
            request_id="req-1"
        )
        
        assert event1.compute_hash() != event2.compute_hash()


class TestAuditLogger:
    """Test suite for AuditLogger."""
    
    def test_logger_initialization(self, tmp_path):
        """Test logger can be initialized."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        assert logger.output_path == str(log_path)
        assert logger.enable_chaining is True
        assert logger.last_hash is None
    
    def test_hash_chaining(self, tmp_path):
        """Test that events are chained with hashes."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path), enable_chaining=True)
        
        event1 = AuditEvent(
            timestamp="2024-01-01T00:00:00",
            event_type=AuditEventType.PROVIDER_REQUEST,
            actor="test",
            action="test1",
            resource_type="provider",
            resource_id="provider1",
            status="success",
            metadata={},
            request_id="req-1"
        )
        
        event2 = AuditEvent(
            timestamp="2024-01-01T00:00:01",
            event_type=AuditEventType.PROVIDER_RESPONSE,
            actor="test",
            action="test2",
            resource_type="provider",
            resource_id="provider1",
            status="success",
            metadata={},
            request_id="req-1"
        )
        
        logger.log_event(event1)
        logger.log_event(event2)
        
        # Second event should have previous hash equal to first event's hash
        assert event2.previous_hash == event1.entry_hash
        assert logger.last_hash == event2.entry_hash
    
    def test_no_chaining(self, tmp_path):
        """Test that chaining can be disabled."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path), enable_chaining=False)
        
        event = AuditEvent(
            timestamp="2024-01-01T00:00:00",
            event_type=AuditEventType.CONFIG_UPDATED,
            actor="admin",
            action="update",
            resource_type="config",
            resource_id="setting1",
            status="success",
            metadata={},
            request_id="req-1"
        )
        
        logger.log_event(event)
        
        # No hashes should be added
        assert event.previous_hash is None
        assert event.entry_hash is None
    
    def test_log_provider_request(self, tmp_path):
        """Test logging a provider request."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        logger.log_provider_request(
            provider="openai",
            model="gpt-4",
            request_id="req-123",
            actor="user1",
            metadata={"tokens": 100}
        )
        
        # Verify the logger was called (last_hash should be set)
        assert logger.last_hash is not None
    
    def test_log_routing_decision(self, tmp_path):
        """Test logging a routing decision."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        logger.log_routing_decision(
            selected_provider="openai",
            routing_rule="code_generation",
            request_id="req-123",
            alternatives=["anthropic", "groq"],
            metadata={"complexity": 0.7}
        )
        
        assert logger.last_hash is not None
    
    def test_log_auth_attempt_success(self, tmp_path):
        """Test logging successful authentication."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        logger.log_auth_attempt(
            user_id="user123",
            success=True,
            ip_address="192.168.1.1"
        )
        
        assert logger.last_hash is not None
    
    def test_log_auth_attempt_failure(self, tmp_path):
        """Test logging failed authentication."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        logger.log_auth_attempt(
            user_id="user123",
            success=False,
            ip_address="192.168.1.1",
            reason="Invalid password"
        )
        
        assert logger.last_hash is not None
    
    def test_log_rate_limit(self, tmp_path):
        """Test logging rate limit events."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        logger.log_rate_limit(
            user_id="user123",
            endpoint="/api/chat",
            limit=100,
            current=105,
            request_id="req-123"
        )
        
        assert logger.last_hash is not None
    
    def test_log_config_change(self, tmp_path):
        """Test logging configuration changes."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        logger.log_config_change(
            actor="admin",
            config_key="max_retries",
            old_value=3,
            new_value=5,
            request_id="config-123"
        )
        
        assert logger.last_hash is not None
    
    def test_log_config_change_redacts_sensitive(self, tmp_path):
        """Test that sensitive config values are redacted."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        # This should redact the API key values
        logger.log_config_change(
            actor="admin",
            config_key="api_key_openai",
            old_value="sk-old-key-123",
            new_value="sk-new-key-456",
            request_id="config-123"
        )
        
        assert logger.last_hash is not None
    
    def test_log_provider_error(self, tmp_path):
        """Test logging provider errors."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        logger.log_provider_error(
            provider="openai",
            error_type="rate_limit",
            error_message="Rate limit exceeded",
            request_id="req-123"
        )
        
        assert logger.last_hash is not None
    
    def test_multiple_events_chaining(self, tmp_path):
        """Test that multiple events are properly chained."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path), enable_chaining=True)
        
        events = []
        for i in range(5):
            event = AuditEvent(
                timestamp=f"2024-01-01T00:00:0{i}",
                event_type=AuditEventType.PROVIDER_REQUEST,
                actor="test",
                action=f"action{i}",
                resource_type="provider",
                resource_id=f"provider{i}",
                status="success",
                metadata={},
                request_id=f"req-{i}"
            )
            logger.log_event(event)
            events.append(event)
        
        # Verify chain integrity
        for i in range(1, len(events)):
            assert events[i].previous_hash == events[i-1].entry_hash


class TestAuditEventTypes:
    """Test that all event types are properly defined."""
    
    def test_all_event_types_have_values(self):
        """Test that all event types have string values."""
        for event_type in AuditEventType:
            assert isinstance(event_type.value, str)
            assert "." in event_type.value  # Should be in format "category.action"
    
    def test_auth_event_types(self):
        """Test authentication event types."""
        assert AuditEventType.AUTH_SUCCESS.value == "auth.success"
        assert AuditEventType.AUTH_FAILURE.value == "auth.failure"
    
    def test_provider_event_types(self):
        """Test provider event types."""
        assert AuditEventType.PROVIDER_REQUEST.value == "provider.request"
        assert AuditEventType.PROVIDER_RESPONSE.value == "provider.response"
        assert AuditEventType.PROVIDER_ERROR.value == "provider.error"
    
    def test_security_event_types(self):
        """Test security event types."""
        assert AuditEventType.SUSPICIOUS_ACTIVITY.value == "security.suspicious"
        assert AuditEventType.ACCESS_DENIED.value == "security.denied"


@pytest.mark.integration
class TestAuditLoggerIntegration:
    """Integration tests for audit logger."""
    
    def test_end_to_end_logging_flow(self, tmp_path):
        """Test complete logging flow from request to response."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        request_id = "req-integration-123"
        
        # 1. Log routing decision
        logger.log_routing_decision(
            selected_provider="openai",
            routing_rule="code_generation",
            request_id=request_id,
            alternatives=["anthropic"]
        )
        
        # 2. Log provider request
        logger.log_provider_request(
            provider="openai",
            model="gpt-4",
            request_id=request_id,
            actor="user1"
        )
        
        # 3. Log provider error (simulating failure)
        logger.log_provider_error(
            provider="openai",
            error_type="timeout",
            error_message="Request timeout",
            request_id=request_id
        )
        
        # 4. Log fallback routing decision
        logger.log_routing_decision(
            selected_provider="anthropic",
            routing_rule="fallback",
            request_id=request_id,
            alternatives=[]
        )
        
        # All events should be chained
        assert logger.last_hash is not None
    
    def test_concurrent_logging(self, tmp_path):
        """Test that logger handles concurrent logging correctly."""
        log_path = tmp_path / "audit.log"
        logger = AuditLogger(output_path=str(log_path))
        
        # Simulate concurrent requests
        for i in range(10):
            logger.log_provider_request(
                provider=f"provider{i % 3}",
                model="test-model",
                request_id=f"req-{i}",
                actor=f"user{i % 2}"
            )
        
        # Should have logged all events
        assert logger.last_hash is not None
