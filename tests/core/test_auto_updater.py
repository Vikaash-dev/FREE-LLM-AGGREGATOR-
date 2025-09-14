import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from datetime import datetime

from src.core.auto_updater import AutoUpdater, UpdateSource, ProviderUpdate
from src.models import ModelInfo

class TestAutoUpdater(unittest.TestCase):
    def test_initialization(self):
        """Test that the AutoUpdater can be initialized."""
        with patch.object(AutoUpdater, '_load_update_sources', return_value=[]), \
             patch.object(AutoUpdater, '_load_cache', return_value={}):
            mock_account_manager = MagicMock()
            updater = AutoUpdater(account_manager=mock_account_manager)
            self.assertIsNotNone(updater)


class TestAutoUpdaterFunctionality(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_account_manager = MagicMock()
        # Patch the file I/O methods to avoid actual file system access
        self.load_sources_patcher = patch.object(AutoUpdater, '_load_update_sources', return_value=[])
        self.load_cache_patcher = patch.object(AutoUpdater, '_load_cache', return_value={})
        self.save_cache_patcher = patch.object(AutoUpdater, '_save_cache', return_value=None)

        self.mock_load_sources = self.load_sources_patcher.start()
        self.mock_load_cache = self.load_cache_patcher.start()
        self.mock_save_cache = self.save_cache_patcher.start()

        self.updater = AutoUpdater(account_manager=self.mock_account_manager)

    def tearDown(self):
        self.load_sources_patcher.stop()
        self.load_cache_patcher.stop()
        self.save_cache_patcher.stop()

    async def test_check_for_updates_calls_correct_method_for_api_source(self):
        """Verify that check_for_updates dispatches to the correct internal method for 'api' type."""
        # Arrange
        api_source = UpdateSource(name="test_api", type="api", url="http://fakeapi.com", update_interval=1)
        self.updater.sources = [api_source]

        # Patch the internal method that would be called
        self.updater._update_from_api = AsyncMock(return_value=[])

        # Act
        await self.updater.check_for_updates()

        # Assert
        self.updater._update_from_api.assert_called_once_with(api_source)

    async def test_check_for_updates_respects_update_interval(self):
        """Verify that check_for_updates respects the update interval."""
        # Arrange
        source = UpdateSource(
            name="test_api", type="api", url="http://fakeapi.com",
            update_interval=24, # 24 hours
            last_updated=datetime.now() # Just updated
        )
        self.updater.sources = [source]
        self.updater._update_from_api = AsyncMock()

        # Act
        await self.updater.check_for_updates()

        # Assert
        self.updater._update_from_api.assert_not_called()

    def test_compare_model_lists(self):
        """Test the logic for comparing cached and new model lists."""
        # Arrange
        cached_models = [
            ModelInfo(name="model-1", display_name="Model One", context_length=4096, is_free=True, provider="p", capabilities=[]),
            ModelInfo(name="model-2", display_name="Model Two", context_length=8192, is_free=False, provider="p", capabilities=[]),
            ModelInfo(name="model-to-be-removed", display_name="Old Model", context_length=2048, is_free=True, provider="p", capabilities=[]),
        ]

        new_models = [
            ModelInfo(name="model-1", display_name="Model One v2", context_length=4096, is_free=True, provider="p", capabilities=[]), # Updated
            ModelInfo(name="model-2", display_name="Model Two", context_length=8192, is_free=False, provider="p", capabilities=[]), # Unchanged
            ModelInfo(name="model-3", display_name="New Model", context_length=16000, is_free=True, provider="p", capabilities=[]), # Added
        ]

        # Act
        added, removed, updated = self.updater._compare_model_lists(
            [m.model_dump() for m in cached_models], new_models
        )

        # Assert
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].name, "model-3")

        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0], "model-to-be-removed")

        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0].name, "model-1")
        self.assertEqual(updated[0].display_name, "Model One v2")


if __name__ == '__main__':
    unittest.main()
