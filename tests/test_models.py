"""Unit tests for tvbingefriend_recommendation_service.models."""
import pytest
from datetime import datetime, UTC
from tvbingefriend_recommendation_service.models.show_metdata import ShowMetadata
from tvbingefriend_recommendation_service.models.show_similarity import ShowSimilarity
from tvbingefriend_recommendation_service.models.base import Base


class TestShowMetadata:
    """Tests for ShowMetadata model."""

    def test_show_metadata_creation(self, test_db_session):
        """Test creating a ShowMetadata record."""
        # Arrange & Act
        show = ShowMetadata(
            show_id=1,
            name='Test Show',
            genres=['Drama', 'Crime'],
            summary='A test show',
            rating=8.5,
            type='Scripted',
            language='English',
            network='TestNet'
        )
        test_db_session.add(show)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowMetadata).filter_by(show_id=1).first()
        assert retrieved is not None
        assert retrieved.name == 'Test Show'
        assert retrieved.genres == ['Drama', 'Crime']
        assert retrieved.rating == 8.5

    def test_show_metadata_repr(self):
        """Test __repr__ method."""
        # Arrange
        show = ShowMetadata(show_id=1, name='Test Show')

        # Act
        repr_str = repr(show)

        # Assert
        assert 'ShowMetadata' in repr_str
        assert 'show_id=1' in repr_str
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


class TestShowSimilarity:
    """Tests for ShowSimilarity model."""

    def test_show_similarity_creation(self, test_db_session):
        """Test creating a ShowSimilarity record."""
        # Arrange & Act
        similarity = ShowSimilarity(
            show_id=1,
            similar_show_id=2,
            similarity_score=0.85,
            genre_score=0.9,
            text_score=0.8,
            metadata_score=0.85
        )
        test_db_session.add(similarity)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowSimilarity).filter_by(
            show_id=1, similar_show_id=2
        ).first()
        assert retrieved is not None
        assert retrieved.similarity_score == 0.85
        assert retrieved.genre_score == 0.9

    def test_show_similarity_repr(self):
        """Test __repr__ method."""
        # Arrange
        similarity = ShowSimilarity(
            show_id=1,
            similar_show_id=2,
            similarity_score=0.85
        )

        # Act
        repr_str = repr(similarity)

        # Assert
        assert 'ShowSimilarity' in repr_str
        assert 'show_id=1' in repr_str
        assert 'similar_show_id=2' in repr_str
        assert '0.850' in repr_str

    def test_show_similarity_composite_primary_key(self, test_db_session):
        """Test that show_id and similar_show_id form composite primary key."""
        # Arrange
        similarity1 = ShowSimilarity(
            show_id=1, similar_show_id=2, similarity_score=0.8
        )
        similarity2 = ShowSimilarity(
            show_id=1, similar_show_id=3, similarity_score=0.7
        )
        test_db_session.add_all([similarity1, similarity2])
        test_db_session.commit()

        # Act
        results = test_db_session.query(ShowSimilarity).filter_by(show_id=1).all()

        # Assert
        assert len(results) == 2

    def test_show_similarity_computed_at_default(self, test_db_session):
        """Test that computed_at has a default value."""
        # Arrange & Act
        similarity = ShowSimilarity(
            show_id=1, similar_show_id=2, similarity_score=0.8
        )
        test_db_session.add(similarity)
        test_db_session.commit()

        # Assert
        assert similarity.computed_at is not None
        assert isinstance(similarity.computed_at, datetime)


class TestBase:
    """Tests for Base declarative base."""

    def test_base_is_declarative_base(self):
        """Test that Base is a declarative base."""
        # Assert
        assert hasattr(Base, 'metadata')
        assert hasattr(Base, 'registry')


class TestDatabase:
    """Tests for database module."""

    def test_database_url_validation(self, monkeypatch):
        """Test that database raises error with no URL."""
        # Arrange
        monkeypatch.delenv('DATABASE_URL', raising=False)

        # Act & Assert
        from unittest.mock import patch
        import sys
        import importlib

        # Remove the module from cache so it can be reimported
        if 'tvbingefriend_recommendation_service.models.database' in sys.modules:
            del sys.modules['tvbingefriend_recommendation_service.models.database']

        # Mock get_database_url to return None before import
        with patch('tvbingefriend_recommendation_service.config.get_database_url', return_value=None):
            with pytest.raises(ValueError, match="DATABASE_URL is not configured"):
                import tvbingefriend_recommendation_service.models.database as new_db_module
