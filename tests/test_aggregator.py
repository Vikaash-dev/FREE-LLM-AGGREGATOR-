"""
Tests for the LLM Aggregator core functionality.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, UTC

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    AccountCredentials,
    ProviderConfig,
    ModelInfo,
    ModelCapability,
    ProviderType,
    ProviderStatus,
    RateLimit
)
from src.core.aggregator import LLMAggregator
from src.core.account_manager import AccountManager
from src.core.router import ProviderRouter
from src.core.rate_limiter import RateLimiter
from src.core.state_tracker import StateTracker
from src.providers.base import BaseProvider, ProviderError


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(self, name: str, should_fail: bool = False):
        # 1. Create the config first
        config = ProviderConfig(
            name=name,
            display_name=name.title(),
            provider_type=ProviderType.FREE,
            base_url=f"https://api.{name}.com",
            models=[
                ModelInfo(
                    name=f"{name}-model-1",
                    display_name=f"{name.title()} Model 1",
                    provider=name,
                    capabilities=[ModelCapability.TEXT_GENERATION],
                    context_length=4096,
                    is_free=True
                )
            ],
            rate_limit=RateLimit(),
            priority=1,
            status=ProviderStatus.ACTIVE
        )

        # 2. Create mock credentials
        credentials = [
            AccountCredentials(
                provider=name,
                account_id="test-account",
                api_key="test-key"
            )
        ]

        # 3. Call super().__init__
        super().__init__(config=config, credentials=credentials)

        # 4. Set mock-specific attributes
        self.should_fail = should_fail
    
    async def chat_completion(self, request: ChatCompletionRequest, credentials: AccountCredentials) -> ChatCompletionResponse:
        if self.should_fail:
            raise ProviderError("Mock provider error")
        
        return ChatCompletionResponse(
            id="test-id",
            object="chat.completion",
            created=int(datetime.now(UTC).timestamp()),
            model=request.model,
            provider=self.name,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Mock response from {self.name}"
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        )
    
    async def chat_completion_stream(self, request: ChatCompletionRequest, credentials: AccountCredentials):
        if self.should_fail:
            raise ProviderError("Mock provider error")
        
        # Yield a single chunk
        yield ChatCompletionResponse(
            id="test-id",
            object="chat.completion.chunk",
            created=int(datetime.now(UTC).timestamp()),
            model=request.model,
            provider=self.name,
            choices=[{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": f"Mock stream from {self.name}"
                },
                "finish_reason": None
            }]
        )
    
    def list_models(self):
        return self.config.models
    
    async def health_check(self) -> bool:
        return not self.should_fail
    
    async def close(self):
        pass


@pytest.fixture
def mock_account_manager():
    """Create a mock account manager."""
    manager = AsyncMock(spec=AccountManager)
    
    # Mock credentials
    credentials = AccountCredentials(
        provider="test-provider",
        account_id="test-account",
        api_key="test-key",
        is_active=True,
        created_at=datetime.now(UTC)
    )
    
    manager.get_credentials.return_value = credentials
    manager.update_usage.return_value = None
    manager.mark_credentials_invalid.return_value = None
    
    return manager


@pytest.fixture
def mock_router():
    """Create a mock router."""
    router = AsyncMock(spec=ProviderRouter)
    router.get_provider_chain.return_value = ["provider1", "provider2"]
    return router


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter."""
    limiter = AsyncMock(spec=RateLimiter)
    limiter.acquire.return_value = True
    limiter.release.return_value = None
    return limiter


@pytest.fixture
def mock_state_tracker():
    """Create a mock state tracker."""
    return AsyncMock(spec=StateTracker)


@pytest.fixture
def aggregator(mock_account_manager, mock_router, mock_rate_limiter, mock_state_tracker, monkeypatch):
    """Create an aggregator with mock dependencies."""

    # Use an in-memory SQLite database for tests
    monkeypatch.setattr("src.config.settings.DATABASE_URL", ":memory:")
    
    providers = [
        MockProvider("provider1"),
        MockProvider("provider2")
    ]
    
    aggregator = LLMAggregator(
        providers=providers,
        account_manager=mock_account_manager,
        router=mock_router,
        rate_limiter=mock_rate_limiter,
        state_tracker=mock_state_tracker,
        enable_auto_updater=False # Disable background task for tests
    )
    
    return aggregator


@pytest.mark.asyncio
async def test_chat_completion_success(aggregator):
    """Test successful chat completion."""
    
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hello")]
    )
    
    response = await aggregator.chat_completion(request)
    
    assert response is not None
    assert response.provider == "provider1"
    assert response.choices[0].message.content == "Mock response from provider1"


@pytest.mark.asyncio
async def test_chat_completion_fallback(aggregator):
    """Test fallback to second provider when first fails."""
    
    # Make first provider fail
    aggregator.providers["provider1"].should_fail = True
    
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hello")]
    )
    
    response = await aggregator.chat_completion(request)
    
    assert response is not None
    assert response.provider == "provider2"
    assert response.choices[0].message.content == "Mock response from provider2"


@pytest.mark.asyncio
async def test_chat_completion_all_providers_fail(aggregator):
    """Test when all providers fail."""
    
    # Make all providers fail
    aggregator.providers["provider1"].should_fail = True
    aggregator.providers["provider2"].should_fail = True
    
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hello")]
    )
    
    with pytest.raises(ProviderError):
        await aggregator.chat_completion(request)


@pytest.mark.asyncio
async def test_chat_completion_stream(aggregator):
    """Test streaming chat completion."""
    
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hello")],
        stream=True
    )
    
    chunks = []
    async for chunk in aggregator.chat_completion_stream(request):
        chunks.append(chunk)
    
    assert len(chunks) == 1
    assert chunks[0].provider == "provider1"
    assert chunks[0].choices[0].delta.content == "Mock stream from provider1"


@pytest.mark.asyncio
async def test_specific_provider_request(aggregator):
    """Test request with specific provider."""
    
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hello")],
        provider="provider2"
    )
    
    response = await aggregator.chat_completion(request)
    
    assert response.provider == "provider2"


@pytest.mark.asyncio
async def test_rate_limiting(aggregator, mock_rate_limiter):
    """Test rate limiting integration."""
    
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hello")]
    )
    
    await aggregator.chat_completion(request, user_id="test-user")
    
    # Verify rate limiter was called
    mock_rate_limiter.acquire.assert_called_once_with("test-user")
    mock_rate_limiter.release.assert_called_once_with("test-user")


@pytest.mark.asyncio
async def test_credentials_management(aggregator, mock_account_manager):
    """Test credentials management integration."""
    
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hello")]
    )
    
    await aggregator.chat_completion(request)
    
    # Verify credentials were requested and usage updated
    mock_account_manager.get_credentials.assert_called()
    mock_account_manager.update_usage.assert_called()


@pytest.mark.asyncio
async def test_model_selection(aggregator):
    """Test automatic model selection."""
    
    # Test with auto model
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Write some Python code")]
    )
    
    response = await aggregator.chat_completion(request)
    assert response is not None
    
    # The model should be resolved from "auto" to an actual model name
    assert response.model != "auto"


@pytest.mark.asyncio
async def test_list_available_models(aggregator):
    """Test listing available models."""
    
    models = await aggregator.list_available_models()
    
    assert "provider1" in models
    assert "provider2" in models
    assert len(models["provider1"]) == 1
    assert models["provider1"][0].name == "provider1-model-1"


@pytest.mark.asyncio
async def test_provider_status(aggregator):
    """Test getting provider status."""
    
    status = await aggregator.get_provider_status()
    
    assert "provider1" in status
    assert "provider2" in status
    assert status["provider1"]["available"] is True
    assert status["provider1"]["models_count"] == 1


@pytest.mark.asyncio
async def test_health_check(aggregator):
    """Test health check functionality."""
    
    health = await aggregator.health_check()
    
    assert "provider1" in health
    assert "provider2" in health
    assert health["provider1"] is True
    assert health["provider2"] is True


@pytest.mark.asyncio
async def test_health_check_with_failure(aggregator):
    """Test health check when provider fails."""
    
    aggregator.providers["provider1"].should_fail = True
    
    health = await aggregator.health_check()
    
    assert health["provider1"] is False
    assert health["provider2"] is True


@pytest.mark.asyncio
async def test_capability_inference(aggregator):
    """Test capability inference from request content."""
    
    # Test code generation request
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Write a Python function to sort a list")]
    )
    
    capabilities = aggregator._infer_capabilities(request)
    assert ModelCapability.TEXT_GENERATION in capabilities
    assert ModelCapability.CODE_GENERATION in capabilities
    
    # Test reasoning request
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Think about this problem and solve it step by step")]
    )
    
    capabilities = aggregator._infer_capabilities(request)
    assert ModelCapability.TEXT_GENERATION in capabilities
    assert ModelCapability.REASONING in capabilities


if __name__ == "__main__":
    pytest.main([__file__])