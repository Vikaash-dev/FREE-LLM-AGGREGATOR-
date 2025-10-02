#!/usr/bin/env python3
"""
Quick setup script to apply critical fixes
"""

import os
import shutil
import subprocess
import sys


def apply_fixes():
    print("🔧 Applying critical fixes to OpenHands...")
    
    # 1. Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        with open('.env', 'w') as f:
            f.write("""# Security
ADMIN_TOKEN=change-this-secure-token
ALLOWED_ORIGINS=http://localhost:3000

# Database
DATABASE_URL=sqlite:///./model_memory.db
REDIS_URL=redis://localhost:6379

# Encryption
OPENHANDS_ENCRYPTION_KEY=change-this-32-byte-encryption-key

# General Application Settings
LOG_LEVEL=INFO

# LLM Aggregator Settings
MAX_RETRIES=3
RETRY_DELAY=1.0

# Meta Controller Settings
META_CONTROLLER_LEARNING_RATE=0.1
META_CONTROLLER_EXPLORATION_RATE=0.1

# Auto Updater Settings
AUTO_UPDATE_INTERVAL_MINUTES=60
""")
        print("✅ Created .env file - PLEASE UPDATE THE TOKENS!")
    else:
        print("ℹ️  .env file already exists, skipping...")
    
    # 2. Create pytest.ini if it doesn't exist
    if not os.path.exists('pytest.ini'):
        print("📝 Creating pytest.ini...")
        with open('pytest.ini', 'w') as f:
            f.write("""[tool:pytest]
asyncio_default_fixture_loop_scope = function
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
""")
        print("✅ Created pytest.ini")
    else:
        print("ℹ️  pytest.ini already exists, skipping...")
    
    # 3. Remove credentials.json from git tracking (if exists)
    if os.path.exists('credentials.json'):
        print("🔒 Moving credentials.json to credentials.json.backup...")
        shutil.move('credentials.json', 'credentials.json.backup')
        print("✅ Backed up credentials.json")
    
    # 4. Create .gitignore entries (only if not already present)
    gitignore_entries = [
        "",
        "# Security",
        ".env",
        "credentials.json",
        "*.key",
        "*.pem",
        "",
        "# Database",
        "*.db",
        "*.sqlite",
        "",
        "# Logs",
        "*.log",
        "logs/",
        "",
        "# Cache",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        "",
        "# ML Models",
        "models/",
        "checkpoints/",
    ]
    
    # Check if entries already exist
    existing_gitignore = ""
    if os.path.exists('.gitignore'):
        with open('.gitignore', 'r') as f:
            existing_gitignore = f.read()
    
    # Only add entries that don't exist
    entries_to_add = []
    for entry in gitignore_entries:
        if entry and entry not in existing_gitignore:
            entries_to_add.append(entry)
    
    if entries_to_add:
        print("📝 Updating .gitignore...")
        with open('.gitignore', 'a') as f:
            f.write('\n' + '\n'.join(entries_to_add) + '\n')
        print("✅ Updated .gitignore")
    else:
        print("ℹ️  .gitignore already up to date, skipping...")
    
    # 5. Create requirements-ml.txt if it doesn't exist
    if not os.path.exists('requirements-ml.txt'):
        print("📝 Creating requirements-ml.txt...")
        with open('requirements-ml.txt', 'w') as f:
            f.write("""# Optional ML features - install with: pip install -r requirements-ml.txt
# These are not required for basic functionality

torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
transformers>=4.30.0
""")
        print("✅ Created requirements-ml.txt")
    else:
        print("ℹ️  requirements-ml.txt already exists, skipping...")
    
    # 6. Create conftest.py if it doesn't exist
    if not os.path.exists('conftest.py'):
        print("📝 Creating conftest.py...")
        with open('conftest.py', 'w') as f:
            f.write("""\"\"\"
Pytest configuration and fixtures for testing.
\"\"\"

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(scope="session")
def event_loop():
    \"\"\"Create an instance of the default event loop for the test session.\"\"\"
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_aggregator():
    \"\"\"Mock LLM Aggregator for testing.\"\"\"
    aggregator = AsyncMock()
    aggregator.chat_completion.return_value = {
        "id": "test-123",
        "object": "chat.completion",
        "choices": [{"message": {"content": "Test response"}}]
    }
    return aggregator


@pytest.fixture
def mock_provider():
    \"\"\"Mock provider for testing.\"\"\"
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
    \"\"\"Mock credentials for testing.\"\"\"
    return {
        "provider": "test-provider",
        "account_id": "test-account",
        "api_key": "test-key",
        "additional_headers": {}
    }
""")
        print("✅ Created conftest.py")
    else:
        print("ℹ️  conftest.py already exists, skipping...")
    
    print("\n✅ Applied critical fixes!")
    print("\n🚨 IMPORTANT NEXT STEPS:")
    print("1. Update the tokens in .env file before running!")
    print("2. Install dependencies with: pip install -r requirements.txt")
    print("3. OPTIONAL: For ML features, install: pip install -r requirements-ml.txt")
    print("4. Run tests with: python -m pytest")
    print("5. Start the server with: python main.py --port 8000")
    print("\n⚠️  WARNING: Change default tokens in .env to secure values before deploying!")


if __name__ == "__main__":
    apply_fixes()
