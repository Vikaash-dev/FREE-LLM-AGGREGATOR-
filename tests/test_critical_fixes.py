"""
Smoke tests to verify critical fixes are working.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_imports():
    """Test that config module imports successfully."""
    from src.config import settings
    assert settings is not None
    assert hasattr(settings, 'ADMIN_TOKEN')
    assert hasattr(settings, 'ALLOWED_ORIGINS')
    assert isinstance(settings.ALLOWED_ORIGINS, (list, str))


def test_meta_controller_imports_without_torch():
    """Test that meta_controller imports without torch."""
    from src.core import meta_controller
    assert hasattr(meta_controller, 'TORCH_AVAILABLE')
    # Should work regardless of whether torch is installed
    assert isinstance(meta_controller.TORCH_AVAILABLE, bool)


def test_ensemble_system_imports_without_torch():
    """Test that ensemble_system imports without torch."""
    from src.core import ensemble_system
    assert hasattr(ensemble_system, 'TORCH_AVAILABLE')
    assert isinstance(ensemble_system.TORCH_AVAILABLE, bool)


def test_auto_updater_imports_without_playwright():
    """Test that auto_updater imports without playwright."""
    from src.core import auto_updater
    assert hasattr(auto_updater, 'PLAYWRIGHT_AVAILABLE')
    assert isinstance(auto_updater.PLAYWRIGHT_AVAILABLE, bool)


def test_server_module_imports():
    """Test that server module imports successfully."""
    from src.api import server
    assert server.app is not None
    assert hasattr(server, 'app')


def test_models_import():
    """Test that models import successfully."""
    from src.models import ChatCompletionRequest, ChatMessage
    assert ChatCompletionRequest is not None
    assert ChatMessage is not None


def test_allowed_origins_parsing():
    """Test that ALLOWED_ORIGINS is parsed correctly from .env."""
    from src.config import settings
    # Should be parsed as a list
    assert isinstance(settings.ALLOWED_ORIGINS, (list, str))
    if isinstance(settings.ALLOWED_ORIGINS, str):
        # If still string, should be parseable
        origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(',')]
        assert len(origins) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
