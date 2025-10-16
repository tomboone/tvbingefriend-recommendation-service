"""
Tests for scripts/download_from_blob.py
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Import functions from the script
from scripts.download_from_blob import (
    download_processed_data,
    main
)


class TestDownloadProcessedData:
    """Tests for download_processed_data function."""

    @patch('scripts.download_from_blob.BlobStorageClient')
    def test_download_processed_data_success(self, mock_blob_client_class, temp_data_dir):
        """Test successful download of processed data."""
        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client

        # Mock list_blobs to return some files
        mock_blob_client.list_blobs.return_value = [
            'processed/genre_features.npy',
            'processed/text_features.npz',
            'processed/shows_metadata.csv'
        ]

        # Mock download_directory to return count
        mock_blob_client.download_directory.return_value = 3

        result = download_processed_data(
            local_dir=temp_data_dir,
            blob_prefix='processed',
            connection_string='test_connection_string',
            container_name='test-container'
        )

        # Verify BlobStorageClient was initialized
        mock_blob_client_class.assert_called_once_with(
            connection_string='test_connection_string',
            container_name='test-container'
        )

        # Verify list_blobs was called
        mock_blob_client.list_blobs.assert_called_once_with(prefix='processed')

        # Verify download_directory was called
        mock_blob_client.download_directory.assert_called_once_with(
            blob_prefix='processed',
            local_dir=temp_data_dir
        )

        # Verify result
        assert result == 3

    @patch('scripts.download_from_blob.BlobStorageClient')
    @patch('scripts.download_from_blob.get_azure_storage_connection_string')
    @patch('scripts.download_from_blob.get_storage_container_name')
    def test_download_processed_data_uses_config_defaults(
        self, mock_get_container, mock_get_conn, mock_blob_client_class, temp_data_dir
    ):
        """Test that config values are used when not provided."""
        mock_get_conn.return_value = 'config_connection_string'
        mock_get_container.return_value = 'config-container'

        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client
        mock_blob_client.list_blobs.return_value = ['processed/file.npy']
        mock_blob_client.download_directory.return_value = 1

        download_processed_data(
            local_dir=temp_data_dir,
            blob_prefix='processed'
        )

        # Verify config functions were called
        mock_get_conn.assert_called_once()
        mock_get_container.assert_called_once()

        # Verify client was initialized with config values
        mock_blob_client_class.assert_called_once_with(
            connection_string='config_connection_string',
            container_name='config-container'
        )

    @patch('scripts.download_from_blob.get_azure_storage_connection_string')
    def test_download_processed_data_no_connection_string(
        self, mock_get_conn, temp_data_dir
    ):
        """Test handling when connection string is not available."""
        mock_get_conn.return_value = None

        result = download_processed_data(
            local_dir=temp_data_dir,
            blob_prefix='processed'
        )

        # Should return 0 when no connection string
        assert result == 0

    @patch('scripts.download_from_blob.BlobStorageClient')
    def test_download_processed_data_no_blobs_found(
        self, mock_blob_client_class, temp_data_dir
    ):
        """Test when no blobs are found with the given prefix."""
        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client
        mock_blob_client.list_blobs.return_value = []

        result = download_processed_data(
            local_dir=temp_data_dir,
            blob_prefix='nonexistent',
            connection_string='test_conn',
            container_name='test-container'
        )

        # Should return 0 when no blobs found
        assert result == 0

        # download_directory should not be called
        mock_blob_client.download_directory.assert_not_called()

    @patch('scripts.download_from_blob.BlobStorageClient')
    def test_download_processed_data_custom_blob_prefix(
        self, mock_blob_client_class, temp_data_dir
    ):
        """Test download with custom blob prefix."""
        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client
        mock_blob_client.list_blobs.return_value = [
            'custom/prefix/file1.npy',
            'custom/prefix/file2.npz'
        ]
        mock_blob_client.download_directory.return_value = 2

        download_processed_data(
            local_dir=temp_data_dir,
            blob_prefix='custom/prefix',
            connection_string='test_conn',
            container_name='test-container'
        )

        # Verify list_blobs was called with custom prefix
        mock_blob_client.list_blobs.assert_called_once_with(prefix='custom/prefix')

        # Verify download_directory was called with custom prefix
        call_kwargs = mock_blob_client.download_directory.call_args[1]
        assert call_kwargs['blob_prefix'] == 'custom/prefix'

    @patch('scripts.download_from_blob.BlobStorageClient')
    def test_download_processed_data_creates_local_dir(
        self, mock_blob_client_class, temp_data_dir
    ):
        """Test that local directory is passed correctly."""
        local_dir = temp_data_dir / 'downloads'

        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client
        mock_blob_client.list_blobs.return_value = ['processed/file.npy']
        mock_blob_client.download_directory.return_value = 1

        download_processed_data(
            local_dir=local_dir,
            blob_prefix='processed',
            connection_string='test_conn',
            container_name='test-container'
        )

        # Verify download_directory was called with correct local_dir
        call_kwargs = mock_blob_client.download_directory.call_args[1]
        assert call_kwargs['local_dir'] == local_dir

    @patch('scripts.download_from_blob.BlobStorageClient')
    def test_download_processed_data_multiple_files(
        self, mock_blob_client_class, temp_data_dir
    ):
        """Test downloading multiple files."""
        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client
        mock_blob_client.list_blobs.return_value = [
            'processed/genre_features.npy',
            'processed/text_features.npz',
            'processed/platform_features.npy',
            'processed/type_features.npy',
            'processed/language_features.npy',
            'processed/shows_metadata.csv'
        ]
        mock_blob_client.download_directory.return_value = 6

        result = download_processed_data(
            local_dir=temp_data_dir,
            blob_prefix='processed',
            connection_string='test_conn',
            container_name='test-container'
        )

        assert result == 6


class TestMain:
    """Tests for main function."""

    @patch('scripts.download_from_blob.download_processed_data')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_success_with_defaults(
        self, mock_parse_args, mock_download, temp_data_dir
    ):
        """Test main function with default arguments."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = 'processed'
        mock_args.connection_string = None
        mock_args.container_name = None
        mock_parse_args.return_value = mock_args

        mock_download.return_value = 5

        main()

        # Verify download was called
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['blob_prefix'] == 'processed'
        assert call_kwargs['connection_string'] is None
        assert call_kwargs['container_name'] is None

    @patch('scripts.download_from_blob.download_processed_data')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_success_with_custom_args(
        self, mock_parse_args, mock_download, temp_data_dir
    ):
        """Test main function with custom arguments."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = 'custom/path'
        mock_args.connection_string = 'custom_conn_string'
        mock_args.container_name = 'custom-container'
        mock_parse_args.return_value = mock_args

        mock_download.return_value = 3

        main()

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs['blob_prefix'] == 'custom/path'
        assert call_kwargs['connection_string'] == 'custom_conn_string'
        assert call_kwargs['container_name'] == 'custom-container'

    @patch('scripts.download_from_blob.download_processed_data')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_main_no_files_downloaded(
        self, mock_exit, mock_parse_args, mock_download, temp_data_dir
    ):
        """Test main function when no files are downloaded."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = 'processed'
        mock_args.connection_string = None
        mock_args.container_name = None
        mock_parse_args.return_value = mock_args

        mock_download.return_value = 0

        main()

        # Should exit with error when no files downloaded
        mock_exit.assert_called_once_with(1)

    @patch('scripts.download_from_blob.download_processed_data')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_main_handles_download_error(
        self, mock_exit, mock_parse_args, mock_download, temp_data_dir
    ):
        """Test main function handles download errors."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = 'processed'
        mock_args.connection_string = None
        mock_args.container_name = None
        mock_parse_args.return_value = mock_args

        mock_download.side_effect = Exception("Download failed")

        main()

        mock_exit.assert_called_once_with(1)

    @patch('scripts.download_from_blob.download_processed_data')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_successful_download_count(
        self, mock_parse_args, mock_download, temp_data_dir
    ):
        """Test main function with successful downloads."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = 'processed'
        mock_args.connection_string = 'test_conn'
        mock_args.container_name = 'test-container'
        mock_parse_args.return_value = mock_args

        mock_download.return_value = 8

        main()

        # Verify download returned expected count
        assert mock_download.return_value == 8

    @patch('scripts.download_from_blob.download_processed_data')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_with_all_parameters(
        self, mock_parse_args, mock_download, temp_data_dir
    ):
        """Test main function with all parameters specified."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = 'production/data'
        mock_args.connection_string = 'DefaultEndpointsProtocol=https;AccountName=test;'
        mock_args.container_name = 'production-container'
        mock_parse_args.return_value = mock_args

        mock_download.return_value = 10

        main()

        # Verify all parameters were passed
        call_kwargs = mock_download.call_args[1]
        assert 'local_dir' in call_kwargs
        assert call_kwargs['blob_prefix'] == 'production/data'
        assert call_kwargs['connection_string'] == 'DefaultEndpointsProtocol=https;AccountName=test;'
        assert call_kwargs['container_name'] == 'production-container'

    @patch('scripts.download_from_blob.download_processed_data')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_creates_local_directory_path(
        self, mock_parse_args, mock_download, temp_data_dir
    ):
        """Test that local directory path is created correctly."""
        local_path = str(temp_data_dir / 'data' / 'processed')

        mock_args = Mock()
        mock_args.local_dir = local_path
        mock_args.blob_prefix = 'processed'
        mock_args.connection_string = None
        mock_args.container_name = None
        mock_parse_args.return_value = mock_args

        mock_download.return_value = 5

        main()

        # Verify local_dir was passed correctly
        call_kwargs = mock_download.call_args[1]
        assert str(call_kwargs['local_dir']).endswith('data/processed')

    @patch('scripts.download_from_blob.download_processed_data')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_main_connection_error(
        self, mock_exit, mock_parse_args, mock_download, temp_data_dir
    ):
        """Test main function handles connection errors."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = 'processed'
        mock_args.connection_string = 'invalid_connection'
        mock_args.container_name = 'test-container'
        mock_parse_args.return_value = mock_args

        mock_download.side_effect = Exception("Connection failed")

        main()

        mock_exit.assert_called_once_with(1)

    @patch('scripts.download_from_blob.download_processed_data')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_main_empty_blob_storage(
        self, mock_exit, mock_parse_args, mock_download, temp_data_dir
    ):
        """Test main function when blob storage is empty."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = 'processed'
        mock_args.connection_string = 'test_conn'
        mock_args.container_name = 'test-container'
        mock_parse_args.return_value = mock_args

        # Return 0 to indicate no files downloaded
        mock_download.return_value = 0

        main()

        # Should exit with error
        mock_exit.assert_called_once_with(1)
