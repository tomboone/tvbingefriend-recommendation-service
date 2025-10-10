"""Application configuration"""
import os
import json
from pathlib import Path
from typing import Optional


def _get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get configuration value from environment or local.settings.json.

    Priority:
    1. Environment variable
    2. local.settings.json (Values.key)
    3. Default value

    Args:
        key: Configuration key name
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    # Try environment variable first
    value = os.getenv(key)
    if value:
        return value

    # Try local.settings.json
    project_root = Path(__file__).resolve().parent.parent
    local_settings_path = project_root / 'local.settings.json'

    if local_settings_path.exists():
        try:
            with open(local_settings_path) as f:
                settings = json.load(f)
                value = settings.get('Values', {}).get(key)
                if value:
                    return value
        except (json.JSONDecodeError, KeyError):
            pass

    # Return default
    return default


def get_database_url() -> str:
    """
    Get database URL from environment or config.

    Returns:
        Database connection string
    """
    return _get_config_value(
        'DATABASE_URL',
        default='mysql+pymysql://user:password@localhost:3306/tvbingefriend_recommendations'
    )


def get_service_url(service_name: str, default_port: int) -> str:
    """
    Get service URL from config.

    Args:
        service_name: Service name (e.g., 'show', 'season')
        default_port: Default port number

    Returns:
        Service URL
    """
    env_key = f'{service_name.upper()}_SERVICE_URL'
    return _get_config_value(env_key, default=f'http://localhost:{default_port}/api')


def get_azure_storage_connection_string() -> Optional[str]:
    """
    Get Azure Storage connection string from environment or config.

    Returns:
        Connection string or None
    """
    return _get_config_value('AZURE_STORAGE_CONNECTION_STRING')


def get_storage_container_name() -> str:
    """
    Get storage container name from config.

    Returns:
        Container name (default: recommendation-data)
    """
    return _get_config_value('STORAGE_CONTAINER_NAME', default='recommendation-data')


def use_blob_storage() -> bool:
    """
    Check if blob storage should be used (vs local files).

    Returns:
        True if blob storage should be used
    """
    # If connection string is available, use blob storage
    if get_azure_storage_connection_string():
        return True

    # Check explicit flag
    use_blob = _get_config_value('USE_BLOB_STORAGE', default='false')
    return use_blob.lower() == 'true'
