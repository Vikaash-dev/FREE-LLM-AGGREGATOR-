"""
Pytest configuration and fixtures for testing.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_aggregator():
    """Mock LLM Aggregator for testing."""
    aggregator = AsyncMock()
    aggregator.chat_completion.return_value = {
        "id": "test-123",
        "object": "chat.completion",
        "choices": [{"message": {"content": "Test response"}}]
    }
    return aggregator


@pytest.fixture
def mock_provider():
    """Mock provider for testing."""
    provider = AsyncMock()
    provider.name = "test-provider"
    provider.chat_completion.return_value = {
        "id": "test-123",
        "object": "chat.completion",
        "choices": [{"message": {"content": "Test response"}}]
    }
    return provider


@pytest.fixture
def mock_credentials():
    """Mock credentials for testing."""
    return {
        "provider": "test-provider",
        "account_id": "test-account",
        "api_key": "test-key",
        "additional_headers": {}
    }
