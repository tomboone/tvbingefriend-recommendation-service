"""Unit tests for tvbingefriend_recommendation_service.storage.blob_storage."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
import tempfile

from tvbingefriend_recommendation_service.storage.blob_storage import (
    BlobStorageClient,
    get_blob_client
)


class TestBlobStorageClientInit:
    """Tests for BlobStorageClient initialization."""

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_init_with_connection_string_and_container(self, mock_storage_service):
        """Test initialization with explicit parameters."""
        # Arrange
        mock_container_client = Mock()
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        # Act
        client = BlobStorageClient(
            connection_string='test_connection',
            container_name='test-container'
        )

        # Assert
        assert client.container_name == 'test-container'
        assert client.container_client == mock_container_client
        mock_storage_service.assert_called_once_with(connection_string='test_connection')

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    @patch('tvbingefriend_recommendation_service.storage.blob_storage.get_storage_container_name')
    def test_init_with_default_container_name(self, mock_get_container, mock_storage_service):
        """Test initialization with default container name from config."""
        # Arrange
        mock_get_container.return_value = 'default-container'
        mock_container_client = Mock()
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        # Act
        client = BlobStorageClient(connection_string='test_connection')

        # Assert
        assert client.container_name == 'default-container'
        mock_get_container.assert_called_once()


class TestUploadFile:
    """Tests for upload_file method."""

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_upload_file_success(self, mock_storage_service, tmp_path):
        """Test successful file upload."""
        # Arrange
        test_file = tmp_path / 'test.txt'
        test_file.write_text('test content')

        mock_blob_client = Mock()
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.upload_file(test_file, 'test_blob.txt')

        # Assert
        assert result is True
        mock_container_client.get_blob_client.assert_called_once_with('test_blob.txt')
        mock_blob_client.upload_blob.assert_called_once()

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_upload_file_failure(self, mock_storage_service, tmp_path):
        """Test file upload failure."""
        # Arrange
        test_file = tmp_path / 'test.txt'
        test_file.write_text('test content')

        mock_blob_client = Mock()
        mock_blob_client.upload_blob.side_effect = Exception('Upload failed')
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.upload_file(test_file, 'test_blob.txt')

        # Assert
        assert result is False


class TestDownloadFile:
    """Tests for download_file method."""

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_download_file_success(self, mock_storage_service, tmp_path):
        """Test successful file download."""
        # Arrange
        download_path = tmp_path / 'downloaded.txt'

        mock_download_result = Mock()
        mock_download_result.readall.return_value = b'downloaded content'
        mock_blob_client = Mock()
        mock_blob_client.download_blob.return_value = mock_download_result
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.download_file('test_blob.txt', download_path)

        # Assert
        assert result is True
        assert download_path.exists()
        assert download_path.read_bytes() == b'downloaded content'

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_download_file_creates_parent_directory(self, mock_storage_service, tmp_path):
        """Test that download creates parent directory if needed."""
        # Arrange
        download_path = tmp_path / 'subdir' / 'downloaded.txt'

        mock_download_result = Mock()
        mock_download_result.readall.return_value = b'content'
        mock_blob_client = Mock()
        mock_blob_client.download_blob.return_value = mock_download_result
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.download_file('test_blob.txt', download_path)

        # Assert
        assert result is True
        assert download_path.parent.exists()

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_download_file_failure(self, mock_storage_service, tmp_path):
        """Test file download failure."""
        # Arrange
        download_path = tmp_path / 'downloaded.txt'

        mock_blob_client = Mock()
        mock_blob_client.download_blob.side_effect = Exception('Download failed')
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.download_file('test_blob.txt', download_path)

        # Assert
        assert result is False


class TestFileExists:
    """Tests for file_exists method."""

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_file_exists_returns_true_when_exists(self, mock_storage_service):
        """Test file_exists returns True for existing blob."""
        # Arrange
        mock_blob_client = Mock()
        mock_blob_client.exists.return_value = True
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.file_exists('test_blob.txt')

        # Assert
        assert result is True

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_file_exists_returns_false_when_not_exists(self, mock_storage_service):
        """Test file_exists returns False for non-existent blob."""
        # Arrange
        mock_blob_client = Mock()
        mock_blob_client.exists.return_value = False
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.file_exists('test_blob.txt')

        # Assert
        assert result is False

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_file_exists_handles_exception(self, mock_storage_service):
        """Test file_exists handles exceptions gracefully."""
        # Arrange
        mock_blob_client = Mock()
        mock_blob_client.exists.side_effect = Exception('Error')
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.file_exists('test_blob.txt')

        # Assert
        assert result is False


class TestListBlobs:
    """Tests for list_blobs method."""

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_list_blobs_returns_blob_names(self, mock_storage_service):
        """Test listing blobs returns blob names."""
        # Arrange
        mock_blob1 = Mock()
        mock_blob1.name = 'blob1.txt'
        mock_blob2 = Mock()
        mock_blob2.name = 'blob2.txt'

        mock_container_client = Mock()
        mock_container_client.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.list_blobs()

        # Assert
        assert result == ['blob1.txt', 'blob2.txt']

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_list_blobs_with_prefix(self, mock_storage_service):
        """Test listing blobs with prefix filter."""
        # Arrange
        mock_blob = Mock()
        mock_blob.name = 'processed/data.npy'

        mock_container_client = Mock()
        mock_container_client.list_blobs.return_value = [mock_blob]
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.list_blobs(prefix='processed')

        # Assert
        mock_container_client.list_blobs.assert_called_once_with(name_starts_with='processed')
        assert result == ['processed/data.npy']

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_list_blobs_handles_exception(self, mock_storage_service):
        """Test list_blobs handles exceptions gracefully."""
        # Arrange
        mock_container_client = Mock()
        mock_container_client.list_blobs.side_effect = Exception('Error')
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.list_blobs()

        # Assert
        assert result == []


class TestDeleteFile:
    """Tests for delete_file method."""

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_delete_file_success(self, mock_storage_service):
        """Test successful file deletion."""
        # Arrange
        mock_blob_client = Mock()
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.delete_file('test_blob.txt')

        # Assert
        assert result is True
        mock_blob_client.delete_blob.assert_called_once()

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_delete_file_failure(self, mock_storage_service):
        """Test file deletion failure."""
        # Arrange
        mock_blob_client = Mock()
        mock_blob_client.delete_blob.side_effect = Exception('Delete failed')
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        result = client.delete_file('test_blob.txt')

        # Assert
        assert result is False


class TestUploadDirectory:
    """Tests for upload_directory method."""

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_upload_directory_uploads_all_files(self, mock_storage_service, tmp_path):
        """Test uploading all files from directory."""
        # Arrange
        (tmp_path / 'file1.txt').write_text('content1')
        (tmp_path / 'file2.txt').write_text('content2')

        mock_blob_client = Mock()
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        count = client.upload_directory(tmp_path)

        # Assert
        assert count == 2

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_upload_directory_with_blob_prefix(self, mock_storage_service, tmp_path):
        """Test uploading directory with blob prefix."""
        # Arrange
        (tmp_path / 'file1.txt').write_text('content1')

        mock_blob_client = Mock()
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        count = client.upload_directory(tmp_path, blob_prefix='processed')

        # Assert
        assert count == 1
        mock_container_client.get_blob_client.assert_called_with('processed/file1.txt')

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_upload_directory_with_file_patterns(self, mock_storage_service, tmp_path):
        """Test uploading directory with file pattern filter."""
        # Arrange
        (tmp_path / 'file1.txt').write_text('content1')
        (tmp_path / 'file2.csv').write_text('content2')
        (tmp_path / 'file3.txt').write_text('content3')

        mock_blob_client = Mock()
        mock_container_client = Mock()
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        count = client.upload_directory(tmp_path, file_patterns=['*.txt'])

        # Assert
        assert count == 2  # Only .txt files

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_upload_directory_nonexistent_returns_zero(self, mock_storage_service, tmp_path):
        """Test uploading non-existent directory returns 0."""
        # Arrange
        nonexistent_dir = tmp_path / 'nonexistent'

        mock_container_client = Mock()
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        count = client.upload_directory(nonexistent_dir)

        # Assert
        assert count == 0


class TestDownloadDirectory:
    """Tests for download_directory method."""

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_download_directory_downloads_all_blobs(self, mock_storage_service, tmp_path):
        """Test downloading all blobs with prefix."""
        # Arrange
        mock_blob1 = Mock()
        mock_blob1.name = 'processed/file1.txt'
        mock_blob2 = Mock()
        mock_blob2.name = 'processed/file2.txt'

        mock_download_result = Mock()
        mock_download_result.readall.return_value = b'content'

        mock_blob_client = Mock()
        mock_blob_client.download_blob.return_value = mock_download_result

        mock_container_client = Mock()
        mock_container_client.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_container_client.get_blob_client.return_value = mock_blob_client
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        count = client.download_directory('processed', tmp_path)

        # Assert
        assert count == 2
        assert (tmp_path / 'file1.txt').exists()
        assert (tmp_path / 'file2.txt').exists()

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.StorageService')
    def test_download_directory_creates_local_dir(self, mock_storage_service, tmp_path):
        """Test that download creates local directory."""
        # Arrange
        download_dir = tmp_path / 'downloads'

        mock_container_client = Mock()
        mock_container_client.list_blobs.return_value = []
        mock_storage_service.return_value.get_blob_service_client.return_value = mock_container_client

        client = BlobStorageClient('test_conn', 'test-container')

        # Act
        client.download_directory('processed', download_dir)

        # Assert
        assert download_dir.exists()


class TestGetBlobClient:
    """Tests for get_blob_client factory function."""

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.BlobStorageClient')
    def test_get_blob_client_with_explicit_params(self, mock_client_class):
        """Test factory function with explicit parameters."""
        # Act
        get_blob_client(
            connection_string='test_conn',
            container_name='test-container'
        )

        # Assert
        mock_client_class.assert_called_once_with(
            connection_string='test_conn',
            container_name='test-container'
        )

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.BlobStorageClient')
    def test_get_blob_client_uses_env_variable(self, mock_client_class, monkeypatch):
        """Test factory function uses environment variable."""
        # Arrange
        monkeypatch.setenv('AZURE_STORAGE_CONNECTION_STRING', 'env_connection')

        # Act
        get_blob_client()

        # Assert
        mock_client_class.assert_called_once_with(
            connection_string='env_connection',
            container_name=None
        )

    @patch('tvbingefriend_recommendation_service.storage.blob_storage.BlobStorageClient')
    def test_get_blob_client_with_none_connection_string(self, mock_client_class, monkeypatch):
        """Test factory function when no connection string available."""
        # Arrange
        monkeypatch.delenv('AZURE_STORAGE_CONNECTION_STRING', raising=False)

        # Act
        get_blob_client()

        # Assert
        mock_client_class.assert_called_once_with(
            connection_string=None,
            container_name=None
        )
