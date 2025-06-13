#!/usr/bin/env python3
"""
Health check script for OpenHands LLM API Aggregator.
"""

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import httpx
from src.config.settings import settings


async def check_health():
    """Check the health of the aggregator."""
    
    try:
        async with httpx.AsyncClient() as client:
            # Check main health endpoint
            response = await client.get(f"http://{settings.HOST}:{settings.PORT}/health")
            
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server is healthy: {health_data.get('status', 'unknown')}")
                
                # Check providers
                providers_response = await client.get(f"http://{settings.HOST}:{settings.PORT}/v1/models")
                if providers_response.status_code == 200:
                    models = providers_response.json()
                    print(f"📡 Available models: {len(models.get('data', []))}")
                
                return True
            else:
                print(f"❌ Server unhealthy: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


if __name__ == "__main__":
    result = asyncio.run(check_health())
    sys.exit(0 if result else 1)
