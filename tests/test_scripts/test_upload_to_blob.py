"""
Tests for scripts/upload_to_blob.py
"""

from unittest.mock import Mock, patch

# Import functions from the script
from scripts.upload_to_blob import main, upload_processed_data


class TestUploadProcessedData:
    """Tests for upload_processed_data function."""

    @patch("scripts.upload_to_blob.BlobStorageClient")
    def test_upload_processed_data_success(self, mock_blob_client_class, temp_data_dir):
        """Test successful upload of processed data."""
        # Create test files
        test_files = [
            "genre_features.npy",
            "text_features.npz",
            "platform_features.npy",
            "type_features.npy",
            "language_features.npy",
            "tfidf_vectorizer.pkl",
            "genre_encoder.pkl",
            "shows_metadata.csv",
        ]
        for filename in test_files:
            (temp_data_dir / filename).touch()

        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client
        mock_blob_client.upload_file.return_value = True
        mock_blob_client.list_blobs.return_value = [f"processed/{f}" for f in test_files]

        result = upload_processed_data(
            local_dir=temp_data_dir,
            blob_prefix="processed",
            connection_string="test_connection_string",
            container_name="test-container",
        )

        # Verify BlobStorageClient was initialized
        mock_blob_client_class.assert_called_once_with(
            connection_string="test_connection_string", container_name="test-container"
        )

        # Verify all files were uploaded
        assert result == len(test_files)
        assert mock_blob_client.upload_file.call_count == len(test_files)

        # Verify list_blobs was called
        mock_blob_client.list_blobs.assert_called_once_with(prefix="processed")

    @patch("scripts.upload_to_blob.BlobStorageClient")
    @patch("scripts.upload_to_blob.get_azure_storage_connection_string")
    @patch("scripts.upload_to_blob.get_storage_container_name")
    def test_upload_processed_data_uses_config_defaults(
        self, mock_get_container, mock_get_conn, mock_blob_client_class, temp_data_dir
    ):
        """Test that config values are used when not provided."""
        # Create at least one test file
        (temp_data_dir / "genre_features.npy").touch()

        mock_get_conn.return_value = "config_connection_string"
        mock_get_container.return_value = "config-container"

        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client
        mock_blob_client.upload_file.return_value = True
        mock_blob_client.list_blobs.return_value = []

        upload_processed_data(local_dir=temp_data_dir, blob_prefix="processed")

        # Verify config functions were called
        mock_get_conn.assert_called_once()
        mock_get_container.assert_called_once()

        # Verify client was initialized with config values
        mock_blob_client_class.assert_called_once_with(
            connection_string="config_connection_string", container_name="config-container"
        )

    @patch("scripts.upload_to_blob.get_azure_storage_connection_string")
    def test_upload_processed_data_no_connection_string(self, mock_get_conn, temp_data_dir):
        """Test handling when connection string is not available."""
        mock_get_conn.return_value = None

        result = upload_processed_data(local_dir=temp_data_dir, blob_prefix="processed")

        # Should return 0 when no connection string
        assert result == 0

    @patch("scripts.upload_to_blob.BlobStorageClient")
    def test_upload_processed_data_some_files_missing(self, mock_blob_client_class, temp_data_dir):
        """Test upload when some files are missing."""
        # Create only some files
        (temp_data_dir / "genre_features.npy").touch()
        (temp_data_dir / "text_features.npz").touch()
        # Other files are missing

        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client
        mock_blob_client.upload_file.return_value = True
        mock_blob_client.list_blobs.return_value = []

        result = upload_processed_data(
            local_dir=temp_data_dir,
            blob_prefix="processed",
            connection_string="test_conn",
            container_name="test-container",
        )

        # Only 2 files should be uploaded
        assert result == 2
        assert mock_blob_client.upload_file.call_count == 2

    @patch("scripts.upload_to_blob.BlobStorageClient")
    def test_upload_processed_data_upload_failures(self, mock_blob_client_class, temp_data_dir):
        """Test handling of upload failures."""
        # Create test files
        (temp_data_dir / "genre_features.npy").touch()
        (temp_data_dir / "text_features.npz").touch()

        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client

        # First upload succeeds, second fails
        mock_blob_client.upload_file.side_effect = [True, False]
        mock_blob_client.list_blobs.return_value = []

        result = upload_processed_data(
            local_dir=temp_data_dir,
            blob_prefix="processed",
            connection_string="test_conn",
            container_name="test-container",
        )

        # Only 1 successful upload
        assert result == 1

    @patch("scripts.upload_to_blob.BlobStorageClient")
    def test_upload_processed_data_custom_blob_prefix(self, mock_blob_client_class, temp_data_dir):
        """Test upload with custom blob prefix."""
        (temp_data_dir / "genre_features.npy").touch()

        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client
        mock_blob_client.upload_file.return_value = True
        mock_blob_client.list_blobs.return_value = ["custom/prefix/genre_features.npy"]

        upload_processed_data(
            local_dir=temp_data_dir,
            blob_prefix="custom/prefix",
            connection_string="test_conn",
            container_name="test-container",
        )

        # Verify upload was called with custom prefix
        call_args = mock_blob_client.upload_file.call_args_list[0]
        blob_name = call_args[0][1]
        assert blob_name.startswith("custom/prefix/")

    @patch("scripts.upload_to_blob.BlobStorageClient")
    def test_upload_processed_data_no_files_exist(self, mock_blob_client_class, temp_data_dir):
        """Test when no expected files exist."""
        mock_blob_client = Mock()
        mock_blob_client_class.return_value = mock_blob_client
        mock_blob_client.list_blobs.return_value = []

        result = upload_processed_data(
            local_dir=temp_data_dir,
            blob_prefix="processed",
            connection_string="test_conn",
            container_name="test-container",
        )

        # No files should be uploaded
        assert result == 0
        assert mock_blob_client.upload_file.call_count == 0


class TestMain:
    """Tests for main function."""

    @patch("scripts.upload_to_blob.upload_processed_data")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_success_with_defaults(self, mock_parse_args, mock_upload, temp_data_dir):
        """Test main function with default arguments."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = "processed"
        mock_args.connection_string = None
        mock_args.container_name = None
        mock_parse_args.return_value = mock_args

        mock_upload.return_value = 8

        main()

        # Verify upload was called
        mock_upload.assert_called_once()
        call_kwargs = mock_upload.call_args[1]
        assert call_kwargs["blob_prefix"] == "processed"
        assert call_kwargs["connection_string"] is None
        assert call_kwargs["container_name"] is None

    @patch("scripts.upload_to_blob.upload_processed_data")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_success_with_custom_args(self, mock_parse_args, mock_upload, temp_data_dir):
        """Test main function with custom arguments."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = "custom/path"
        mock_args.connection_string = "custom_conn_string"
        mock_args.container_name = "custom-container"
        mock_parse_args.return_value = mock_args

        mock_upload.return_value = 5

        main()

        mock_upload.assert_called_once()
        call_kwargs = mock_upload.call_args[1]
        assert call_kwargs["blob_prefix"] == "custom/path"
        assert call_kwargs["connection_string"] == "custom_conn_string"
        assert call_kwargs["container_name"] == "custom-container"

    @patch("argparse.ArgumentParser.parse_args")
    @patch("sys.exit")
    def test_main_local_directory_not_found(self, mock_exit, mock_parse_args):
        """Test main function when local directory doesn't exist."""
        mock_args = Mock()
        mock_args.local_dir = "nonexistent/directory"
        mock_args.blob_prefix = "processed"
        mock_args.connection_string = None
        mock_args.container_name = None
        mock_parse_args.return_value = mock_args

        main()

        # Should exit with error code (may be called multiple times)
        assert mock_exit.called
        assert 1 in [call[0][0] for call in mock_exit.call_args_list]

    @patch("scripts.upload_to_blob.upload_processed_data")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("sys.exit")
    def test_main_no_files_uploaded(self, mock_exit, mock_parse_args, mock_upload, temp_data_dir):
        """Test main function when no files are uploaded."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = "processed"
        mock_args.connection_string = None
        mock_args.container_name = None
        mock_parse_args.return_value = mock_args

        mock_upload.return_value = 0

        main()

        # Should exit with error when no files uploaded
        mock_exit.assert_called_once_with(1)

    @patch("scripts.upload_to_blob.upload_processed_data")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("sys.exit")
    def test_main_handles_upload_error(
        self, mock_exit, mock_parse_args, mock_upload, temp_data_dir
    ):
        """Test main function handles upload errors."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = "processed"
        mock_args.connection_string = None
        mock_args.container_name = None
        mock_parse_args.return_value = mock_args

        mock_upload.side_effect = Exception("Upload failed")

        main()

        mock_exit.assert_called_once_with(1)

    @patch("scripts.upload_to_blob.upload_processed_data")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_successful_upload_count(self, mock_parse_args, mock_upload, temp_data_dir):
        """Test main function with successful uploads."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = "processed"
        mock_args.connection_string = "test_conn"
        mock_args.container_name = "test-container"
        mock_parse_args.return_value = mock_args

        mock_upload.return_value = 8

        main()

        # Verify upload returned expected count
        assert mock_upload.return_value == 8

    @patch("scripts.upload_to_blob.upload_processed_data")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_with_all_parameters(self, mock_parse_args, mock_upload, temp_data_dir):
        """Test main function with all parameters specified."""
        mock_args = Mock()
        mock_args.local_dir = str(temp_data_dir)
        mock_args.blob_prefix = "production/data"
        mock_args.connection_string = "DefaultEndpointsProtocol=https;AccountName=test;"
        mock_args.container_name = "production-container"
        mock_parse_args.return_value = mock_args

        mock_upload.return_value = 10

        main()

        # Verify all parameters were passed
        call_kwargs = mock_upload.call_args[1]
        assert "local_dir" in call_kwargs
        assert call_kwargs["blob_prefix"] == "production/data"
        assert (
            call_kwargs["connection_string"] == "DefaultEndpointsProtocol=https;AccountName=test;"
        )
        assert call_kwargs["container_name"] == "production-container"
