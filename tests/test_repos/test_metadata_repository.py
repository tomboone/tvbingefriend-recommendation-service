"""Unit tests for tvbingefriend_recommendation_service.repos.metadata_repository."""

from datetime import UTC, datetime

from tvbingefriend_recommendation_service.models.show_metdata import ShowMetadata
from tvbingefriend_recommendation_service.repos.metadata_repository import MetadataRepository


class TestMetadataRepositoryInit:
    """Tests for MetadataRepository initialization."""

    def test_init_with_session(self, test_db_session):
        """Test initialization with database session."""
        # Act
        repo = MetadataRepository(test_db_session)

        # Assert
        assert repo.db == test_db_session


class TestStoreShow:
    """Tests for store_show method."""

    def test_store_show_creates_new_record(self, metadata_repository, test_db_session):
        """Test storing a new show."""
        # Arrange
        show_data = {
            "show_id": 1,
            "name": "Breaking Bad",
            "genres": ["Drama", "Crime"],
            "summary": "A chemistry teacher cooks meth.",
            "rating": 9.5,
            "type": "Scripted",
            "language": "English",
            "network": "AMC",
        }

        # Act
        result = metadata_repository.store_show(show_data)

        # Assert
        assert result.show_id == 1
        assert result.name == "Breaking Bad"
        assert result.genres == ["Drama", "Crime"]

    def test_store_show_updates_existing_record(self, metadata_repository, test_db_session):
        """Test updating an existing show."""
        # Arrange
        existing_show = ShowMetadata(show_id=1, name="Old Name", rating=7.0)
        test_db_session.add(existing_show)
        test_db_session.commit()

        show_data = {"show_id": 1, "name": "New Name", "rating": 9.0}

        # Act
        result = metadata_repository.store_show(show_data)

        # Assert
        assert result.show_id == 1
        assert result.name == "New Name"
        assert result.rating == 9.0

        # Verify only one record exists
        count = test_db_session.query(ShowMetadata).filter_by(show_id=1).count()
        assert count == 1

    def test_store_show_updates_synced_at(self, metadata_repository, test_db_session):
        """Test that store_show updates synced_at timestamp."""
        # Arrange
        show_data = {"show_id": 1, "name": "Test Show"}

        # Act
        before = datetime.now(UTC)
        result = metadata_repository.store_show(show_data)
        after = datetime.now(UTC)

        # Assert
        assert result.synced_at is not None
        assert isinstance(result.synced_at, datetime)
        # Convert to timestamps for comparison to handle timezone differences
        assert (
            before.timestamp()
            <= result.synced_at.replace(tzinfo=UTC).timestamp()
            <= after.timestamp() + 1
        )

    def test_store_show_with_none_values(self, metadata_repository):
        """Test storing show with None values."""
        # Arrange
        show_data = {"show_id": 1, "name": "Show", "genres": None, "summary": None, "rating": None}

        # Act
        result = metadata_repository.store_show(show_data)

        # Assert
        assert result.genres is None
        assert result.summary is None
        assert result.rating is None


class TestBulkStoreShows:
    """Tests for bulk_store_shows method."""

    def test_bulk_store_shows_creates_multiple_records(self, metadata_repository, test_db_session):
        """Test bulk storing multiple shows."""
        # Arrange
        shows_data = [
            {"show_id": 1, "name": "Show 1", "rating": 8.0},
            {"show_id": 2, "name": "Show 2", "rating": 7.5},
            {"show_id": 3, "name": "Show 3", "rating": 9.0},
        ]

        # Act
        count = metadata_repository.bulk_store_shows(shows_data)

        # Assert
        assert count == 3
        assert test_db_session.query(ShowMetadata).count() == 3

    def test_bulk_store_shows_clears_existing_data(self, metadata_repository, test_db_session):
        """Test that bulk store clears existing data."""
        # Arrange
        existing = ShowMetadata(show_id=99, name="Existing")
        test_db_session.add(existing)
        test_db_session.commit()

        shows_data = [{"show_id": 1, "name": "Show 1"}]

        # Act
        metadata_repository.bulk_store_shows(shows_data)

        # Assert
        # Old data should be gone
        old = test_db_session.query(ShowMetadata).filter_by(show_id=99).first()
        assert old is None

        # New data should exist
        assert test_db_session.query(ShowMetadata).count() == 1

    def test_bulk_store_shows_with_batch_size(self, metadata_repository, test_db_session):
        """Test bulk storing with custom batch size."""
        # Arrange
        shows_data = [{"show_id": i, "name": f"Show {i}"} for i in range(1, 11)]

        # Act
        count = metadata_repository.bulk_store_shows(shows_data, batch_size=3)

        # Assert
        assert count == 10
        assert test_db_session.query(ShowMetadata).count() == 10

    def test_bulk_store_shows_returns_correct_count(self, metadata_repository):
        """Test that bulk store returns correct count."""
        # Arrange
        shows_data = [{"show_id": i, "name": f"Show {i}"} for i in range(1, 6)]

        # Act
        count = metadata_repository.bulk_store_shows(shows_data)

        # Assert
        assert count == 5


class TestGetShow:
    """Tests for get_show method."""

    def test_get_show_returns_existing_show(self, metadata_repository, test_db_session):
        """Test getting an existing show."""
        # Arrange
        show = ShowMetadata(show_id=1, name="Test Show")
        test_db_session.add(show)
        test_db_session.commit()

        # Act
        result = metadata_repository.get_show(1)

        # Assert
        assert result is not None
        assert result.show_id == 1
        assert result.name == "Test Show"

    def test_get_show_returns_none_when_not_found(self, metadata_repository):
        """Test getting a non-existent show."""
        # Act
        result = metadata_repository.get_show(999)

        # Assert
        assert result is None


class TestGetAllShows:
    """Tests for get_all_shows method."""

    def test_get_all_shows_returns_all_records(self, metadata_repository, test_db_session):
        """Test getting all shows."""
        # Arrange
        shows = [
            ShowMetadata(show_id=1, name="Show 1"),
            ShowMetadata(show_id=2, name="Show 2"),
            ShowMetadata(show_id=3, name="Show 3"),
        ]
        test_db_session.add_all(shows)
        test_db_session.commit()

        # Act
        result = metadata_repository.get_all_shows()

        # Assert
        assert len(result) == 3
        assert all(isinstance(show, ShowMetadata) for show in result)

    def test_get_all_shows_returns_empty_list_when_no_data(self, metadata_repository):
        """Test getting all shows when database is empty."""
        # Act
        result = metadata_repository.get_all_shows()

        # Assert
        assert result == []


class TestGetShowIds:
    """Tests for get_show_ids method."""

    def test_get_show_ids_returns_all_ids(self, metadata_repository, test_db_session):
        """Test getting all show IDs."""
        # Arrange
        shows = [
            ShowMetadata(show_id=1, name="Show 1"),
            ShowMetadata(show_id=5, name="Show 5"),
            ShowMetadata(show_id=10, name="Show 10"),
        ]
        test_db_session.add_all(shows)
        test_db_session.commit()

        # Act
        result = metadata_repository.get_show_ids()

        # Assert
        assert len(result) == 3
        assert 1 in result
        assert 5 in result
        assert 10 in result

    def test_get_show_ids_returns_empty_list_when_no_data(self, metadata_repository):
        """Test getting show IDs when database is empty."""
        # Act
        result = metadata_repository.get_show_ids()

        # Assert
        assert result == []


class TestDeleteShow:
    """Tests for delete_show method."""

    def test_delete_show_removes_existing_show(self, metadata_repository, test_db_session):
        """Test deleting an existing show."""
        # Arrange
        show = ShowMetadata(show_id=1, name="Test Show")
        test_db_session.add(show)
        test_db_session.commit()

        # Act
        result = metadata_repository.delete_show(1)

        # Assert
        assert result is True
        assert test_db_session.query(ShowMetadata).filter_by(show_id=1).first() is None

    def test_delete_show_returns_false_when_not_found(self, metadata_repository):
        """Test deleting a non-existent show."""
        # Act
        result = metadata_repository.delete_show(999)

        # Assert
        assert result is False


class TestCountShows:
    """Tests for count_shows method."""

    def test_count_shows_returns_correct_count(self, metadata_repository, test_db_session):
        """Test counting shows."""
        # Arrange
        shows = [ShowMetadata(show_id=i, name=f"Show {i}") for i in range(1, 6)]
        test_db_session.add_all(shows)
        test_db_session.commit()

        # Act
        count = metadata_repository.count_shows()

        # Assert
        assert count == 5

    def test_count_shows_returns_zero_when_empty(self, metadata_repository):
        """Test counting shows when database is empty."""
        # Act
        count = metadata_repository.count_shows()

        # Assert
        assert count == 0
