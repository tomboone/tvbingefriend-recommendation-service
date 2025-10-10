"""Blob storage client wrapper for recommendation data."""
import logging
from pathlib import Path
from typing import Optional, List
import tempfile
import os

from tvbingefriend_azure_storage_service import AzureStorageClient

logger = logging.getLogger(__name__)


class BlobStorageClient:
    """
    Wrapper for Azure Blob Storage operations specific to recommendation data.
    Uses the tvbingefriend-azure-storage-service library.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: str = "recommendation-data"
    ):
        """
        Initialize blob storage client.

        Args:
            connection_string: Azure Storage connection string (from env if None)
            container_name: Container name for recommendation data
        """
        self.container_name = container_name
        self.client = AzureStorageClient(
            connection_string=connection_string,
            container_name=container_name
        )
        logger.info(f"Initialized BlobStorageClient for container: {container_name}")

    def upload_file(self, local_path: Path, blob_name: str) -> bool:
        """
        Upload a file to blob storage.

        Args:
            local_path: Local file path
            blob_name: Blob name (path in storage)

        Returns:
            True if successful
        """
        try:
            logger.info(f"Uploading {local_path} to {blob_name}...")
            self.client.upload_file(str(local_path), blob_name)
            file_size = local_path.stat().st_size / 1024 / 1024
            logger.info(f"✓ Uploaded {blob_name} ({file_size:.2f} MB)")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {blob_name}: {e}")
            return False

    def download_file(self, blob_name: str, local_path: Path) -> bool:
        """
        Download a file from blob storage.

        Args:
            blob_name: Blob name (path in storage)
            local_path: Local destination path

        Returns:
            True if successful
        """
        try:
            logger.info(f"Downloading {blob_name} to {local_path}...")

            # Ensure parent directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self.client.download_file(blob_name, str(local_path))
            file_size = local_path.stat().st_size / 1024 / 1024
            logger.info(f"✓ Downloaded {blob_name} ({file_size:.2f} MB)")
            return True
        except Exception as e:
            logger.error(f"Failed to download {blob_name}: {e}")
            return False

    def file_exists(self, blob_name: str) -> bool:
        """
        Check if a blob exists.

        Args:
            blob_name: Blob name to check

        Returns:
            True if exists
        """
        try:
            return self.client.blob_exists(blob_name)
        except Exception as e:
            logger.error(f"Error checking if {blob_name} exists: {e}")
            return False

    def list_blobs(self, prefix: str = "") -> List[str]:
        """
        List blobs with optional prefix.

        Args:
            prefix: Blob name prefix filter

        Returns:
            List of blob names
        """
        try:
            return self.client.list_blobs(prefix=prefix)
        except Exception as e:
            logger.error(f"Error listing blobs with prefix {prefix}: {e}")
            return []

    def delete_file(self, blob_name: str) -> bool:
        """
        Delete a blob.

        Args:
            blob_name: Blob name to delete

        Returns:
            True if successful
        """
        try:
            logger.info(f"Deleting {blob_name}...")
            self.client.delete_file(blob_name)
            logger.info(f"✓ Deleted {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {blob_name}: {e}")
            return False

    def upload_directory(
        self,
        local_dir: Path,
        blob_prefix: str = "",
        file_patterns: Optional[List[str]] = None
    ) -> int:
        """
        Upload all files from a directory to blob storage.

        Args:
            local_dir: Local directory path
            blob_prefix: Prefix for blob names (like a folder)
            file_patterns: Optional list of file patterns to match (e.g., ['*.npy', '*.csv'])

        Returns:
            Number of files uploaded
        """
        if not local_dir.exists():
            logger.error(f"Directory not found: {local_dir}")
            return 0

        logger.info(f"Uploading directory {local_dir} to blob prefix {blob_prefix}...")

        uploaded_count = 0
        files_to_upload = []

        # Collect files to upload
        if file_patterns:
            for pattern in file_patterns:
                files_to_upload.extend(local_dir.glob(pattern))
        else:
            files_to_upload = [f for f in local_dir.iterdir() if f.is_file()]

        # Upload each file
        for file_path in files_to_upload:
            blob_name = f"{blob_prefix}/{file_path.name}" if blob_prefix else file_path.name
            if self.upload_file(file_path, blob_name):
                uploaded_count += 1

        logger.info(f"✓ Uploaded {uploaded_count} files from {local_dir}")
        return uploaded_count

    def download_directory(
        self,
        blob_prefix: str,
        local_dir: Path
    ) -> int:
        """
        Download all blobs with a prefix to a local directory.

        Args:
            blob_prefix: Blob prefix (like a folder)
            local_dir: Local destination directory

        Returns:
            Number of files downloaded
        """
        logger.info(f"Downloading blobs with prefix {blob_prefix} to {local_dir}...")

        # Create local directory
        local_dir.mkdir(parents=True, exist_ok=True)

        # List all blobs with prefix
        blob_names = self.list_blobs(prefix=blob_prefix)

        downloaded_count = 0
        for blob_name in blob_names:
            # Extract filename from blob name
            filename = Path(blob_name).name
            local_path = local_dir / filename

            if self.download_file(blob_name, local_path):
                downloaded_count += 1

        logger.info(f"✓ Downloaded {downloaded_count} files to {local_dir}")
        return downloaded_count


def get_blob_client(
    connection_string: Optional[str] = None,
    container_name: str = "recommendation-data"
) -> BlobStorageClient:
    """
    Factory function to get a blob storage client.

    Args:
        connection_string: Azure Storage connection string (from env if None)
        container_name: Container name

    Returns:
        BlobStorageClient instance
    """
    # Try to get from environment if not provided
    if connection_string is None:
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    return BlobStorageClient(
        connection_string=connection_string,
        container_name=container_name
    )
