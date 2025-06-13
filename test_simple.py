"""
Simple test to verify the fix works.
"""

import pytest
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.aggregator import LLMAggregator
from src.core.account_manager import AccountManager
from src.core.router import ProviderRouter
from src.core.rate_limiter import RateLimiter
from src.providers.openrouter import create_openrouter_provider


@pytest.mark.asyncio
async def test_basic_initialization():
    """Test that we can initialize the aggregator without errors."""
    
    # Initialize components
    account_manager = AccountManager()
    rate_limiter = RateLimiter()
    
    # Create providers
    providers = []
    openrouter = create_openrouter_provider([])
    providers.append(openrouter)
    
    # Create provider configs dict
    provider_configs = {provider.name: provider.config for provider in providers}
    
    # Initialize router
    router = ProviderRouter(provider_configs)
    
    # Initialize aggregator
    aggregator = LLMAggregator(
        providers=providers,
        account_manager=account_manager,
        router=router,
        rate_limiter=rate_limiter
    )
    
    # Test basic functionality
    assert aggregator is not None
    assert len(aggregator.providers) == 1
    assert aggregator.meta_controller is not None
    assert aggregator.auto_updater is not None
    
    # Cleanup
    await aggregator.close()
    
    print("✅ Basic initialization test passed!")


if __name__ == "__main__":
    asyncio.run(test_basic_initialization())