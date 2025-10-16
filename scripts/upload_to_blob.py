"""
Upload processed data to Azure Blob Storage.
This script uploads feature matrices, similarity matrices, and metadata to blob storage.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging

from tvbingefriend_recommendation_service.config import (
    get_azure_storage_connection_string,
    get_storage_container_name,
)
from tvbingefriend_recommendation_service.storage import BlobStorageClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def upload_processed_data(
    local_dir: Path,
    blob_prefix: str = "processed",
    connection_string: str = None,
    container_name: str = None,
) -> int:
    """
    Upload all processed data files to blob storage.

    Args:
        local_dir: Local directory with processed data
        blob_prefix: Blob prefix (folder) for uploaded files
        connection_string: Azure Storage connection string
        container_name: Storage container name

    Returns:
        Number of files uploaded
    """
    logger.info("=" * 70)
    logger.info("UPLOADING PROCESSED DATA TO BLOB STORAGE")
    logger.info("=" * 70)

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
    logger.info("=" * 70)

    # Initialize blob client
    blob_client = BlobStorageClient(
        connection_string=connection_string, container_name=container_name
    )

    # File patterns to upload (FEATURES ONLY, not similarities)
    file_patterns = [
        "genre_features.npy",
        "text_features.npz",
        "platform_features.npy",
        "type_features.npy",
        "language_features.npy",
        "tfidf_vectorizer.pkl",
        "genre_encoder.pkl",
        "shows_metadata.csv",
    ]

    logger.info(
        "\nUploading features only (excluding similarity matrices for storage optimization)"
    )
    logger.info("Files to upload:")
    for pattern in file_patterns:
        logger.info(f"  - {pattern}")

    # Upload individual files (not all *.npy to avoid similarity matrices)
    uploaded_count = 0
    for filename in file_patterns:
        file_path = local_dir / filename
        if file_path.exists():
            blob_name = f"{blob_prefix}/{filename}"
            if blob_client.upload_file(file_path, blob_name):
                uploaded_count += 1
        else:
            logger.warning(f"⚠️  File not found: {file_path}")

    logger.info("\n" + "=" * 70)
    logger.info("✓ UPLOAD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Uploaded {uploaded_count} files")

    # List uploaded files
    logger.info("\nUploaded files:")
    blobs = blob_client.list_blobs(prefix=blob_prefix)
    for blob_name in blobs:
        logger.info(f"  - {blob_name}")

    return uploaded_count


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Upload processed data to Azure Blob Storage")
    parser.add_argument(
        "--local-dir",
        type=str,
        default="data/processed",
        help="Local directory with processed data (default: data/processed)",
    )
    parser.add_argument(
        "--blob-prefix",
        type=str,
        default="processed",
        help="Blob prefix/folder for uploaded files (default: processed)",
    )
    parser.add_argument(
        "--connection-string",
        type=str,
        default=None,
        help="Azure Storage connection string (default: from config)",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default=None,
        help="Storage container name (default: from config)",
    )

    args = parser.parse_args()

    # Set up paths
    local_dir = project_root / args.local_dir

    if not local_dir.exists():
        logger.error(f"Local directory not found: {local_dir}")
        sys.exit(1)

    try:
        uploaded_count = upload_processed_data(
            local_dir=local_dir,
            blob_prefix=args.blob_prefix,
            connection_string=args.connection_string,
            container_name=args.container_name,
        )

        if uploaded_count == 0:
            logger.warning("No files were uploaded")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during upload: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
