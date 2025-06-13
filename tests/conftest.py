"""
Test configuration and fixtures for the LLM API Aggregator.
"""

import pytest
import asyncio


# Use default pytest-asyncio event loop


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI-compatible response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm doing well, thank you for asking."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }