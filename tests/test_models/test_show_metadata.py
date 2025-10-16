"""Unit tests for tvbingefriend_recommendation_service.models.show_metdata."""
import pytest
from datetime import datetime, UTC
from sqlalchemy.exc import IntegrityError

from tvbingefriend_recommendation_service.models.show_metdata import ShowMetadata


class TestShowMetadata:
    """Tests for ShowMetadata model."""

    def test_show_metadata_creation_with_all_fields(self, test_db_session):
        """Test creating a ShowMetadata record with all fields."""
        # Arrange & Act
        show = ShowMetadata(
            show_id=1,
            name='Breaking Bad',
            genres=['Drama', 'Crime', 'Thriller'],
            summary='A high school chemistry teacher turned methamphetamine producer.',
            rating=9.5,
            type='Scripted',
            language='English',
            network='AMC'
        )
        test_db_session.add(show)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowMetadata).filter_by(show_id=1).first()
        assert retrieved is not None
        assert retrieved.name == 'Breaking Bad'
        assert retrieved.genres == ['Drama', 'Crime', 'Thriller']
        assert retrieved.summary == 'A high school chemistry teacher turned methamphetamine producer.'
        assert retrieved.rating == 9.5
        assert retrieved.type == 'Scripted'
        assert retrieved.language == 'English'
        assert retrieved.network == 'AMC'

    def test_show_metadata_creation_with_minimal_fields(self, test_db_session):
        """Test creating ShowMetadata with only required fields."""
        # Arrange & Act
        show = ShowMetadata(show_id=1, name='Test Show')
        test_db_session.add(show)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowMetadata).filter_by(show_id=1).first()
        assert retrieved is not None
        assert retrieved.show_id == 1
        assert retrieved.name == 'Test Show'
        assert retrieved.genres is None
        assert retrieved.summary is None
        assert retrieved.rating is None

    def test_show_metadata_primary_key_uniqueness(self, test_db_session):
        """Test that show_id is a unique primary key."""
        # Arrange
        show1 = ShowMetadata(show_id=1, name='Show 1')
        test_db_session.add(show1)
        test_db_session.commit()

        # Act & Assert
        show2 = ShowMetadata(show_id=1, name='Show 2')
        test_db_session.add(show2)

        with pytest.warns(match="conflicts with persistent instance"):
            with pytest.raises(IntegrityError):
                test_db_session.commit()

    def test_show_metadata_repr(self):
        """Test __repr__ method."""
        # Arrange
        show = ShowMetadata(show_id=42, name='Test Show')

        # Act
        repr_str = repr(show)

        # Assert
        assert 'ShowMetadata' in repr_str
        assert 'show_id=42' in repr_str
        assert 'Test Show' in repr_str

    def test_show_metadata_synced_at_default(self, test_db_session):
        """Test that synced_at has a default value."""
        # Arrange & Act
        show = ShowMetadata(show_id=1, name='Test')
        test_db_session.add(show)
        test_db_session.commit()

        # Assert
        assert show.synced_at is not None
        assert isinstance(show.synced_at, datetime)
        # Should be recent (within last 5 seconds to account for timezone issues)
        now = datetime.now(UTC)
        time_diff = abs((now - show.synced_at.replace(tzinfo=UTC)).total_seconds())
        assert time_diff < 5, f"synced_at timestamp is too old: {time_diff} seconds"

    def test_show_metadata_with_empty_genres_list(self, test_db_session):
        """Test ShowMetadata with empty genres list."""
        # Arrange & Act
        show = ShowMetadata(show_id=1, name='Show', genres=[])
        test_db_session.add(show)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowMetadata).filter_by(show_id=1).first()
        assert retrieved.genres == []

    def test_show_metadata_with_multiple_genres(self, test_db_session):
        """Test ShowMetadata with multiple genres."""
        # Arrange & Act
        genres = ['Drama', 'Crime', 'Thriller', 'Mystery', 'Suspense']
        show = ShowMetadata(show_id=1, name='Show', genres=genres)
        test_db_session.add(show)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowMetadata).filter_by(show_id=1).first()
        assert retrieved.genres == genres

    def test_show_metadata_with_long_summary(self, test_db_session):
        """Test ShowMetadata with long summary text."""
        # Arrange
        long_summary = 'A' * 5000  # Very long summary
        show = ShowMetadata(show_id=1, name='Show', summary=long_summary)

        # Act
        test_db_session.add(show)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowMetadata).filter_by(show_id=1).first()
        assert retrieved.summary == long_summary

    def test_show_metadata_with_none_values(self, test_db_session):
        """Test ShowMetadata with explicitly None values."""
        # Arrange & Act
        show = ShowMetadata(
            show_id=1,
            name='Show',
            genres=None,
            summary=None,
            rating=None,
            type=None,
            language=None,
            network=None
        )
        test_db_session.add(show)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowMetadata).filter_by(show_id=1).first()
        assert retrieved.genres is None
        assert retrieved.summary is None
        assert retrieved.rating is None
        assert retrieved.type is None
        assert retrieved.language is None
        assert retrieved.network is None

    def test_show_metadata_update(self, test_db_session):
        """Test updating an existing ShowMetadata record."""
        # Arrange
        show = ShowMetadata(show_id=1, name='Original Name', rating=7.5)
        test_db_session.add(show)
        test_db_session.commit()

        # Act
        show.name = 'Updated Name'
        show.rating = 8.5
        show.summary = 'New summary'
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowMetadata).filter_by(show_id=1).first()
        assert retrieved.name == 'Updated Name'
        assert retrieved.rating == 8.5
        assert retrieved.summary == 'New summary'

    def test_show_metadata_delete(self, test_db_session):
        """Test deleting a ShowMetadata record."""
        # Arrange
        show = ShowMetadata(show_id=1, name='Test Show')
        test_db_session.add(show)
        test_db_session.commit()

        # Act
        test_db_session.delete(show)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowMetadata).filter_by(show_id=1).first()
        assert retrieved is None

    def test_show_metadata_query_by_name(self, test_db_session):
        """Test querying ShowMetadata by name."""
        # Arrange
        show1 = ShowMetadata(show_id=1, name='Breaking Bad')
        show2 = ShowMetadata(show_id=2, name='Better Call Saul')
        test_db_session.add_all([show1, show2])
        test_db_session.commit()

        # Act
        retrieved = test_db_session.query(ShowMetadata).filter_by(name='Breaking Bad').first()

        # Assert
        assert retrieved is not None
        assert retrieved.show_id == 1

    def test_show_metadata_count(self, test_db_session):
        """Test counting ShowMetadata records."""
        # Arrange
        shows = [
            ShowMetadata(show_id=1, name='Show 1'),
            ShowMetadata(show_id=2, name='Show 2'),
            ShowMetadata(show_id=3, name='Show 3')
        ]
        test_db_session.add_all(shows)
        test_db_session.commit()

        # Act
        count = test_db_session.query(ShowMetadata).count()

        # Assert
        assert count == 3
