"""Repository for managing cached show metadata."""

import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from tvbingefriend_recommendation_service.models import ShowMetadata

logger = logging.getLogger(__name__)


class MetadataRepository:
    """
    Repository for managing cached show metadata.
    """

    def __init__(self, db: Session):
        self.db = db

    def store_show(self, show_data: dict) -> ShowMetadata:
        """
        Store or update show metadata.

        Args:
            show_data: Dict with show information

        Returns:
            ShowMetadata object
        """
        show_id = show_data["show_id"]

        # Check if exists
        existing = self.db.query(ShowMetadata).filter(ShowMetadata.show_id == show_id).first()

        if existing:
            # Update
            existing.name = show_data["name"]  # type: ignore[assignment]
            existing.genres = show_data.get("genres")  # type: ignore[assignment]
            existing.summary = show_data.get("summary")  # type: ignore[assignment]
            existing.rating = show_data.get("rating")  # type: ignore[assignment]
            existing.type = show_data.get("type")  # type: ignore[assignment]
            existing.language = show_data.get("language")  # type: ignore[assignment]
            existing.network = show_data.get("network")  # type: ignore[assignment]
            existing.synced_at = datetime.now(UTC)  # type: ignore[assignment]
            show = existing
        else:
            # Insert
            show = ShowMetadata(
                show_id=show_id,
                name=show_data["name"],
                genres=show_data.get("genres"),
                summary=show_data.get("summary"),
                rating=show_data.get("rating"),
                type=show_data.get("type"),
                language=show_data.get("language"),
                network=show_data.get("network"),
                synced_at=datetime.now(UTC),
            )
            self.db.add(show)

        self.db.commit()
        self.db.refresh(show)

        return show

    def bulk_store_shows(self, shows_data: list[dict], batch_size: int = 100) -> int:
        """
        Store multiple shows in bulk.

        Args:
            shows_data: List of show data dicts
            batch_size: Batch size for inserts

        Returns:
            Number of shows stored
        """
        # Clear existing data
        logger.info("Clearing existing metadata...")
        self.db.query(ShowMetadata).delete()
        self.db.commit()

        # Prepare records
        records = []
        for show_data in shows_data:
            record = ShowMetadata(
                show_id=show_data["show_id"],
                name=show_data["name"],
                genres=show_data.get("genres"),
                summary=show_data.get("summary"),
                rating=show_data.get("rating"),
                type=show_data.get("type"),
                language=show_data.get("language"),
                network=show_data.get("network"),
                synced_at=datetime.now(UTC),
            )
            records.append(record)

        # Batch insert
        count = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.db.bulk_save_objects(batch)
            self.db.commit()
            count += len(batch)

        logger.info(f"✓ Stored {count} show metadata records")
        return count

    def get_show(self, show_id: int) -> ShowMetadata | None:
        """Get show metadata by ID."""
        return self.db.query(ShowMetadata).filter(ShowMetadata.show_id == show_id).first()

    # noinspection PyTypeChecker
    def get_all_shows(self) -> list[ShowMetadata]:
        """Get all cached show metadata."""
        return self.db.query(ShowMetadata).all()

    # noinspection PyTypeChecker
    def get_show_ids(self) -> list[int]:
        """Get list of all cached show IDs."""
        result = self.db.query(ShowMetadata.show_id).all()
        return [row[0] for row in result]

    def delete_show(self, show_id: int) -> bool:
        """
        Delete show metadata.

        Args:
            show_id: Show ID to delete

        Returns:
            True if deleted, False if not found
        """
        count = self.db.query(ShowMetadata).filter(ShowMetadata.show_id == show_id).delete()
        self.db.commit()

        return count > 0

    def count_shows(self) -> int:
        """Count total cached shows."""
        return self.db.query(ShowMetadata).count()
