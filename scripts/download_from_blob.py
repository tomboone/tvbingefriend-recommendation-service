"""
Download processed data from Azure Blob Storage.
This script downloads feature matrices, similarity matrices, and metadata from blob storage.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging
import argparse

from tvbingefriend_recommendation_service.storage import BlobStorageClient
from tvbingefriend_recommendation_service.config import (
    get_azure_storage_connection_string,
    get_storage_container_name
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def download_processed_data(
    local_dir: Path,
    blob_prefix: str = "processed",
    connection_string: str = None,
    container_name: str = None
) -> int:
    """
    Download all processed data files from blob storage.

    Args:
        local_dir: Local destination directory
        blob_prefix: Blob prefix (folder) to download from
        connection_string: Azure Storage connection string
        container_name: Storage container name

    Returns:
        Number of files downloaded
    """
    logger.info("="*70)
    logger.info("DOWNLOADING PROCESSED DATA FROM BLOB STORAGE")
    logger.info("="*70)

    # Get connection string and container name from config if not provided
    if connection_string is None:
        connection_string = get_azure_storage_connection_string()

    if connection_string is None:
        logger.error("Azure Storage connection string not found!")
        logger.error("Set AZURE_STORAGE_CONNECTION_STRING environment variable")
        return 0

    if container_name is None:
        container_name = get_storage_container_name()

    logger.info(f"Local directory: {local_dir}")
    logger.info(f"Container: {container_name}")
    logger.info(f"Blob prefix: {blob_prefix}")
    logger.info("="*70)

    # Initialize blob client
    blob_client = BlobStorageClient(
        connection_string=connection_string,
        container_name=container_name
    )

    # List available files
    logger.info("\nAvailable files in blob storage:")
    blobs = blob_client.list_blobs(prefix=blob_prefix)
    if not blobs:
        logger.warning(f"No files found with prefix: {blob_prefix}")
        return 0

    for blob_name in blobs:
        logger.info(f"  - {blob_name}")

    # Download directory
    downloaded_count = blob_client.download_directory(
        blob_prefix=blob_prefix,
        local_dir=local_dir
    )

    logger.info("\n" + "="*70)
    logger.info(f"✓ DOWNLOAD COMPLETE")
    logger.info("="*70)
    logger.info(f"Downloaded {downloaded_count} files to {local_dir}")

    return downloaded_count


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Download processed data from Azure Blob Storage'
    )
    parser.add_argument(
        '--local-dir',
        type=str,
        default='data/processed',
        help='Local destination directory (default: data/processed)'
    )
    parser.add_argument(
        '--blob-prefix',
        type=str,
        default='processed',
        help='Blob prefix/folder to download from (default: processed)'
    )
    parser.add_argument(
        '--connection-string',
        type=str,
        default=None,
        help='Azure Storage connection string (default: from config)'
    )
    parser.add_argument(
        '--container-name',
        type=str,
        default=None,
        help='Storage container name (default: from config)'
    )

    args = parser.parse_args()

    # Set up paths
    local_dir = project_root / args.local_dir

    try:
        downloaded_count = download_processed_data(
            local_dir=local_dir,
            blob_prefix=args.blob_prefix,
            connection_string=args.connection_string,
            container_name=args.container_name
        )

        if downloaded_count == 0:
            logger.warning("No files were downloaded")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during download: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
