#!/usr/bin/env python3
"""
Demo script to show the LLM API Aggregator setup and basic functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.aggregator import LLMAggregator
from src.core.account_manager import AccountManager
from src.core.rate_limiter import RateLimiter
from src.core.router import ProviderRouter
from src.providers.openrouter import create_openrouter_provider
from src.providers.groq import create_groq_provider
from src.providers.cerebras import create_cerebras_provider
from src.models import ChatCompletionRequest

async def demo_basic_functionality():
    """Demo basic functionality without API keys."""
    
    print("🚀 LLM API Aggregator Demo")
    print("=" * 50)
    
    # Initialize components
    print("📦 Initializing components...")
    
    try:
        # Initialize components
        account_manager = AccountManager()
        rate_limiter = RateLimiter()
        print("✅ Account Manager and Rate Limiter initialized")
        
        # Create providers
        providers = []
        openrouter = create_openrouter_provider([])
        groq = create_groq_provider([])
        cerebras = create_cerebras_provider([])
        providers.extend([openrouter, groq, cerebras])
        print("✅ Providers created")
        
        # Create provider configs dict
        provider_configs = {provider.name: provider.config for provider in providers}
        
        # Initialize router
        router = ProviderRouter(provider_configs)
        print("✅ Router initialized")
        
        # Initialize aggregator
        aggregator = LLMAggregator(
            providers=providers,
            account_manager=account_manager,
            router=router,
            rate_limiter=rate_limiter
        )
        print("✅ LLM Aggregator initialized")
        print(f"   - Providers loaded: {len(aggregator.providers)}")
        print(f"   - Meta-controller enabled: {aggregator.meta_controller is not None}")
        print(f"   - Auto-updater enabled: {aggregator.auto_updater is not None}")
        
        # List available providers
        print("\n📡 Available Providers:")
        for provider_name, provider in aggregator.providers.items():
            print(f"   - {provider_name}: {provider.__class__.__name__}")
        
        # Show provider status
        print("\n📊 Provider Status:")
        status = await aggregator.get_provider_status()
        for provider_name, provider_status in status.items():
            print(f"   - {provider_name}: {provider_status['status']}")
            if provider_status.get('error'):
                print(f"     Error: {provider_status['error']}")
        
        # List available models (this will show what models are discovered)
        print("\n🤖 Available Models:")
        try:
            models = await aggregator.list_available_models()
            if models:
                for model in models[:5]:  # Show first 5 models
                    print(f"   - {model.id} ({model.provider})")
                if len(models) > 5:
                    print(f"   ... and {len(models) - 5} more models")
            else:
                print("   No models available (API keys not configured)")
        except Exception as e:
            print(f"   Error listing models: {e}")
        
        # Health check
        print("\n🏥 Health Check:")
        try:
            health = await aggregator.health_check()
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Providers: {health.get('providers_available', 0)}/{health.get('total_providers', 0)}")
        except Exception as e:
            print(f"   Health check error: {e}")
        
        # Show auto-updater info
        if aggregator.auto_updater:
            print("\n🔄 Auto-Updater Status:")
            print(f"   - Update sources: {len(aggregator.auto_updater.sources)}")
            print(f"   - Auto-updater running: {hasattr(aggregator.auto_updater, '_update_task')}")
        
        print("\n✅ Demo completed successfully!")
        print("\n💡 Next Steps:")
        print("   1. Run 'python setup.py configure' to add API credentials")
        print("   2. Start the server with 'python main.py'")
        print("   3. Access the web UI at http://localhost:8000")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'aggregator' in locals():
            await aggregator.close()

def main():
    """Main function."""
    asyncio.run(demo_basic_functionality())

if __name__ == "__main__":
    main()