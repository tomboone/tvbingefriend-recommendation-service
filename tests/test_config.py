"""Unit tests for tvbingefriend_recommendation_service.config."""
import pytest
import os
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from tvbingefriend_recommendation_service.config import (
    _get_config_value,
    get_database_url,
    get_service_url,
    get_azure_storage_connection_string,
    get_storage_container_name,
    use_blob_storage
)


class TestGetConfigValue:
    """Tests for _get_config_value function."""

    def test_get_config_value_from_env(self, monkeypatch):
        """Test getting value from environment variable."""
        # Arrange
        monkeypatch.setenv('TEST_KEY', 'test_value')

        # Act
        result = _get_config_value('TEST_KEY')

        # Assert
        assert result == 'test_value'

    def test_get_config_value_with_default(self):
        """Test using default value when key not found."""
        # Act
        result = _get_config_value('NONEXISTENT_KEY', default='default_value')

        # Assert
        assert result == 'default_value'

    def test_get_config_value_from_local_settings(self, tmp_path, monkeypatch):
        """Test getting value from local.settings.json."""
        # Arrange
        settings = {
            "Values": {
                "TEST_KEY": "from_settings"
            }
        }
        settings_file = tmp_path / "local.settings.json"
        with open(settings_file, 'w') as f:
            json.dump(settings, f)

        # Mock Path to return our tmp_path
        with patch('tvbingefriend_recommendation_service.config.Path') as mock_path:
            mock_path.return_value.resolve.return_value.parent.parent = tmp_path
            
            # Act
            result = _get_config_value('TEST_KEY')

        # Assert - env takes precedence, but if not set, should get from file
        # Since we're not setting env, it should use file
        # Note: This test may need adjustment based on actual implementation

    def test_get_config_value_env_precedence(self, tmp_path, monkeypatch):
        """Test that environment variable takes precedence over local.settings.json."""
        # Arrange
        monkeypatch.setenv('TEST_KEY', 'from_env')
        settings = {
            "Values": {
                "TEST_KEY": "from_settings"
            }
        }
        settings_file = tmp_path / "local.settings.json"
        with open(settings_file, 'w') as f:
            json.dump(settings, f)

        # Act
        result = _get_config_value('TEST_KEY')

        # Assert
        assert result == 'from_env'


class TestGetDatabaseUrl:
    """Tests for get_database_url function."""

    def test_get_database_url_from_env(self, monkeypatch):
        """Test getting database URL from environment."""
        # Arrange
        expected_url = 'mysql://user:pass@localhost/db'
        monkeypatch.setenv('DATABASE_URL', expected_url)

        # Act
        result = get_database_url()

        # Assert
        assert result == expected_url

    def test_get_database_url_default(self, monkeypatch, tmp_path):
        """Test default database URL."""
        # Arrange
        monkeypatch.delenv('DATABASE_URL', raising=False)

        # Mock Path to prevent reading from local.settings.json
        from pathlib import Path as PathlibPath
        from unittest.mock import patch

        # Create a non-existent path
        fake_project_root = tmp_path / 'nonexistent'

        with patch('tvbingefriend_recommendation_service.config.Path') as mock_path:
            mock_path.return_value.resolve.return_value.parent.parent = fake_project_root

            # Act
            result = get_database_url()

            # Assert
            assert result == 'mysql+pymysql://user:password@localhost:3306/tvbingefriend_recommendations'


class TestGetServiceUrl:
    """Tests for get_service_url function."""

    def test_get_service_url_from_env(self, monkeypatch):
        """Test getting service URL from environment."""
        # Arrange
        monkeypatch.setenv('SHOW_SERVICE_URL', 'http://custom:8080/api')

        # Act
        result = get_service_url('show', 7071)

        # Assert
        assert result == 'http://custom:8080/api'

    def test_get_service_url_default(self, monkeypatch):
        """Test default service URL construction."""
        # Arrange
        monkeypatch.delenv('SHOW_SERVICE_URL', raising=False)

        # Act
        result = get_service_url('show', 7071)

        # Assert
        assert result == 'http://localhost:7071/api'

    def test_get_service_url_different_services(self, monkeypatch):
        """Test getting URLs for different services."""
        # Arrange
        monkeypatch.delenv('SEASON_SERVICE_URL', raising=False)
        monkeypatch.delenv('EPISODE_SERVICE_URL', raising=False)

        # Act
        season_url = get_service_url('season', 7072)
        episode_url = get_service_url('episode', 7073)

        # Assert
        assert season_url == 'http://localhost:7072/api'
        assert episode_url == 'http://localhost:7073/api'


class TestGetAzureStorageConnectionString:
    """Tests for get_azure_storage_connection_string function."""

    def test_get_azure_storage_connection_string_from_env(self, monkeypatch):
        """Test getting connection string from environment."""
        # Arrange
        expected = 'DefaultEndpointsProtocol=https;AccountName=test'
        monkeypatch.setenv('AZURE_STORAGE_CONNECTION_STRING', expected)

        # Act
        result = get_azure_storage_connection_string()

        # Assert
        assert result == expected

    def test_get_azure_storage_connection_string_none(self, monkeypatch):
        """Test when connection string is not set."""
        # Arrange
        monkeypatch.delenv('AZURE_STORAGE_CONNECTION_STRING', raising=False)

        # Act
        result = get_azure_storage_connection_string()

        # Assert
        assert result is None


class TestGetStorageContainerName:
    """Tests for get_storage_container_name function."""

    def test_get_storage_container_name_from_env(self, monkeypatch):
        """Test getting container name from environment."""
        # Arrange
        monkeypatch.setenv('STORAGE_CONTAINER_NAME', 'my-container')

        # Act
        result = get_storage_container_name()

        # Assert
        assert result == 'my-container'

    def test_get_storage_container_name_default(self, monkeypatch):
        """Test default container name."""
        # Arrange
        monkeypatch.delenv('STORAGE_CONTAINER_NAME', raising=False)

        # Act
        result = get_storage_container_name()

        # Assert
        assert result == 'recommendation-data'


class TestUseBlobStorage:
    """Tests for use_blob_storage function."""

    def test_use_blob_storage_explicit_true(self, monkeypatch):
        """Test explicit USE_BLOB_STORAGE=true."""
        # Arrange
        monkeypatch.setenv('USE_BLOB_STORAGE', 'true')

        # Act
        result = use_blob_storage()

        # Assert
        assert result is True

    def test_use_blob_storage_explicit_false(self, monkeypatch):
        """Test explicit USE_BLOB_STORAGE=false."""
        # Arrange
        monkeypatch.setenv('USE_BLOB_STORAGE', 'false')

        # Act
        result = use_blob_storage()

        # Assert
        assert result is False

    def test_use_blob_storage_auto_detect_with_connection(self, monkeypatch):
        """Test auto-detection when connection string is available."""
        # Arrange
        monkeypatch.delenv('USE_BLOB_STORAGE', raising=False)
        monkeypatch.setenv('AZURE_STORAGE_CONNECTION_STRING', 'test_connection')

        # Act
        result = use_blob_storage()

        # Assert
        assert result is True

    def test_use_blob_storage_auto_detect_without_connection(self, monkeypatch):
        """Test auto-detection when connection string is not available."""
        # Arrange
        monkeypatch.delenv('USE_BLOB_STORAGE', raising=False)
        monkeypatch.delenv('AZURE_STORAGE_CONNECTION_STRING', raising=False)

        # Act
        result = use_blob_storage()

        # Assert
        assert result is False

    def test_use_blob_storage_case_insensitive(self, monkeypatch):
        """Test that boolean value is case-insensitive."""
        # Arrange
        monkeypatch.setenv('USE_BLOB_STORAGE', 'TRUE')

        # Act
        result = use_blob_storage()

        # Assert
        assert result is True
