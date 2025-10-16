"""Unit tests for tvbingefriend_recommendation_service.models.show_similarity."""
import pytest
from datetime import datetime, UTC
from sqlalchemy.exc import IntegrityError

from tvbingefriend_recommendation_service.models.show_similarity import ShowSimilarity


class TestShowSimilarity:
    """Tests for ShowSimilarity model."""

    def test_show_similarity_creation_with_all_fields(self, test_db_session):
        """Test creating a ShowSimilarity record with all fields."""
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
        assert retrieved.text_score == 0.8
        assert retrieved.metadata_score == 0.85

    def test_show_similarity_creation_with_minimal_fields(self, test_db_session):
        """Test creating ShowSimilarity with only required fields."""
        # Arrange & Act
        similarity = ShowSimilarity(
            show_id=1,
            similar_show_id=2,
            similarity_score=0.75
        )
        test_db_session.add(similarity)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowSimilarity).filter_by(
            show_id=1, similar_show_id=2
        ).first()
        assert retrieved is not None
        assert retrieved.similarity_score == 0.75
        assert retrieved.genre_score is None
        assert retrieved.text_score is None
        assert retrieved.metadata_score is None

    def test_show_similarity_composite_primary_key(self, test_db_session):
        """Test that show_id and similar_show_id form composite primary key."""
        # Arrange
        similarity1 = ShowSimilarity(
            show_id=1, similar_show_id=2, similarity_score=0.8
        )
        similarity2 = ShowSimilarity(
            show_id=1, similar_show_id=3, similarity_score=0.7
        )
        similarity3 = ShowSimilarity(
            show_id=2, similar_show_id=1, similarity_score=0.6
        )

        # Act
        test_db_session.add_all([similarity1, similarity2, similarity3])
        test_db_session.commit()

        # Assert
        results = test_db_session.query(ShowSimilarity).all()
        assert len(results) == 3

    def test_show_similarity_duplicate_composite_key_raises_error(self, test_db_session):
        """Test that duplicate composite key raises IntegrityError."""
        # Arrange
        similarity1 = ShowSimilarity(
            show_id=1, similar_show_id=2, similarity_score=0.8
        )
        test_db_session.add(similarity1)
        test_db_session.commit()

        # Act & Assert
        similarity2 = ShowSimilarity(
            show_id=1, similar_show_id=2, similarity_score=0.9
        )
        test_db_session.add(similarity2)

        with pytest.warns(match="conflicts with persistent instance"):
            with pytest.raises(IntegrityError):
                test_db_session.commit()

    def test_show_similarity_repr(self):
        """Test __repr__ method."""
        # Arrange
        similarity = ShowSimilarity(
            show_id=1,
            similar_show_id=2,
            similarity_score=0.856
        )

        # Act
        repr_str = repr(similarity)

        # Assert
        assert 'ShowSimilarity' in repr_str
        assert 'show_id=1' in repr_str
        assert 'similar_show_id=2' in repr_str
        assert '0.856' in repr_str

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
        # Should be recent (within last 5 seconds to account for timezone issues)
        now = datetime.now(UTC)
        time_diff = abs((now - similarity.computed_at.replace(tzinfo=UTC)).total_seconds())
        assert time_diff < 5, f"computed_at timestamp is too old: {time_diff} seconds"

    def test_show_similarity_query_by_show_id(self, test_db_session):
        """Test querying similarities by show_id."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.8),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.7),
            ShowSimilarity(show_id=2, similar_show_id=1, similarity_score=0.6)
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        results = test_db_session.query(ShowSimilarity).filter_by(show_id=1).all()

        # Assert
        assert len(results) == 2
        assert all(sim.show_id == 1 for sim in results)

    def test_show_similarity_order_by_score(self, test_db_session):
        """Test ordering similarities by score."""
        # Arrange
        from sqlalchemy import desc
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.8),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.9),
            ShowSimilarity(show_id=1, similar_show_id=4, similarity_score=0.7)
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        results = (
            test_db_session.query(ShowSimilarity)
            .filter_by(show_id=1)
            .order_by(desc(ShowSimilarity.similarity_score))
            .all()
        )

        # Assert
        assert len(results) == 3
        assert results[0].similarity_score == 0.9
        assert results[1].similarity_score == 0.8
        assert results[2].similarity_score == 0.7

    def test_show_similarity_score_bounds(self, test_db_session):
        """Test that similarity scores can be between 0 and 1."""
        # Arrange & Act
        sim_low = ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.0)
        sim_high = ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=1.0)
        test_db_session.add_all([sim_low, sim_high])
        test_db_session.commit()

        # Assert
        retrieved_low = test_db_session.query(ShowSimilarity).filter_by(
            show_id=1, similar_show_id=2
        ).first()
        retrieved_high = test_db_session.query(ShowSimilarity).filter_by(
            show_id=1, similar_show_id=3
        ).first()

        assert retrieved_low.similarity_score == 0.0
        assert retrieved_high.similarity_score == 1.0

    def test_show_similarity_update(self, test_db_session):
        """Test updating a ShowSimilarity record."""
        # Arrange
        similarity = ShowSimilarity(
            show_id=1, similar_show_id=2, similarity_score=0.7
        )
        test_db_session.add(similarity)
        test_db_session.commit()

        # Act
        similarity.similarity_score = 0.9
        similarity.genre_score = 0.95
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowSimilarity).filter_by(
            show_id=1, similar_show_id=2
        ).first()
        assert retrieved.similarity_score == 0.9
        assert retrieved.genre_score == 0.95

    def test_show_similarity_delete(self, test_db_session):
        """Test deleting a ShowSimilarity record."""
        # Arrange
        similarity = ShowSimilarity(
            show_id=1, similar_show_id=2, similarity_score=0.8
        )
        test_db_session.add(similarity)
        test_db_session.commit()

        # Act
        test_db_session.delete(similarity)
        test_db_session.commit()

        # Assert
        retrieved = test_db_session.query(ShowSimilarity).filter_by(
            show_id=1, similar_show_id=2
        ).first()
        assert retrieved is None

    def test_show_similarity_with_all_score_components(self, test_db_session):
        """Test ShowSimilarity with all score components."""
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
        assert retrieved.genre_score == 0.9
        assert retrieved.text_score == 0.8
        assert retrieved.metadata_score == 0.85

    def test_show_similarity_count(self, test_db_session):
        """Test counting ShowSimilarity records."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.8),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.7),
            ShowSimilarity(show_id=2, similar_show_id=1, similarity_score=0.9)
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        count = test_db_session.query(ShowSimilarity).count()

        # Assert
        assert count == 3
