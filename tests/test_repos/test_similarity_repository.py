"""Unit tests for tvbingefriend_recommendation_service.repos.similarity_repository."""

from datetime import datetime

from tvbingefriend_recommendation_service.models.show_similarity import ShowSimilarity
from tvbingefriend_recommendation_service.repos.similarity_repository import SimilarityRepository


class TestSimilarityRepositoryInit:
    """Tests for SimilarityRepository initialization."""

    def test_init_with_session(self, test_db_session):
        """Test initialization with database session."""
        # Act
        repo = SimilarityRepository(test_db_session)

        # Assert
        assert repo.db == test_db_session


class TestStoreSimilarities:
    """Tests for store_similarities method."""

    def test_store_similarities_creates_new_records(self, similarity_repository, test_db_session):
        """Test storing similarities for a show."""
        # Arrange
        similar_shows = [
            {
                "similar_show_id": 2,
                "similarity_score": 0.85,
                "genre_score": 0.9,
                "text_score": 0.8,
                "metadata_score": 0.85,
            },
            {
                "similar_show_id": 3,
                "similarity_score": 0.75,
                "genre_score": 0.7,
                "text_score": 0.8,
                "metadata_score": 0.75,
            },
        ]

        # Act
        count = similarity_repository.store_similarities(1, similar_shows)

        # Assert
        assert count == 2
        assert test_db_session.query(ShowSimilarity).filter_by(show_id=1).count() == 2

    def test_store_similarities_deletes_existing_records(
        self, similarity_repository, test_db_session
    ):
        """Test that storing similarities deletes existing ones first."""
        # Arrange
        existing = ShowSimilarity(show_id=1, similar_show_id=99, similarity_score=0.5)
        test_db_session.add(existing)
        test_db_session.commit()

        similar_shows = [{"similar_show_id": 2, "similarity_score": 0.8}]

        # Act
        similarity_repository.store_similarities(1, similar_shows)

        # Assert
        # Old record should be gone
        old = test_db_session.query(ShowSimilarity).filter_by(show_id=1, similar_show_id=99).first()
        assert old is None

        # New record should exist
        new = test_db_session.query(ShowSimilarity).filter_by(show_id=1, similar_show_id=2).first()
        assert new is not None

    def test_store_similarities_with_batch_size(self, similarity_repository, test_db_session):
        """Test storing similarities with custom batch size."""
        # Arrange
        similar_shows = [{"similar_show_id": i, "similarity_score": 0.8} for i in range(2, 12)]

        # Act
        count = similarity_repository.store_similarities(1, similar_shows, batch_size=3)

        # Assert
        assert count == 10

    def test_store_similarities_without_optional_scores(
        self, similarity_repository, test_db_session
    ):
        """Test storing similarities without optional score fields."""
        # Arrange
        similar_shows = [{"similar_show_id": 2, "similarity_score": 0.8}]

        # Act
        count = similarity_repository.store_similarities(1, similar_shows)

        # Assert
        assert count == 1
        record = test_db_session.query(ShowSimilarity).filter_by(show_id=1).first()
        assert record.similarity_score == 0.8
        assert record.genre_score is None
        assert record.text_score is None


class TestBulkStoreAllSimilarities:
    """Tests for bulk_store_all_similarities method."""

    def test_bulk_store_all_similarities_creates_all_records(
        self, similarity_repository, test_db_session
    ):
        """Test bulk storing similarities for multiple shows."""
        # Arrange
        all_similarities = {
            1: [
                {"similar_show_id": 2, "similarity_score": 0.85},
                {"similar_show_id": 3, "similarity_score": 0.75},
            ],
            2: [
                {"similar_show_id": 1, "similarity_score": 0.85},
                {"similar_show_id": 3, "similarity_score": 0.65},
            ],
        }

        # Act
        count = similarity_repository.bulk_store_all_similarities(all_similarities)

        # Assert
        assert count == 4
        assert test_db_session.query(ShowSimilarity).count() == 4

    def test_bulk_store_all_similarities_clears_existing_when_requested(
        self, similarity_repository, test_db_session
    ):
        """Test that bulk store clears existing data when requested."""
        # Arrange
        existing = ShowSimilarity(show_id=99, similar_show_id=98, similarity_score=0.5)
        test_db_session.add(existing)
        test_db_session.commit()

        all_similarities = {1: [{"similar_show_id": 2, "similarity_score": 0.8}]}

        # Act
        similarity_repository.bulk_store_all_similarities(all_similarities, clear_existing=True)

        # Assert
        old = test_db_session.query(ShowSimilarity).filter_by(show_id=99).first()
        assert old is None

    def test_bulk_store_all_similarities_preserves_existing_when_not_clearing(
        self, similarity_repository, test_db_session
    ):
        """Test that bulk store preserves existing data when not clearing."""
        # Arrange
        existing = ShowSimilarity(show_id=99, similar_show_id=98, similarity_score=0.5)
        test_db_session.add(existing)
        test_db_session.commit()

        all_similarities = {1: [{"similar_show_id": 2, "similarity_score": 0.8}]}

        # Act
        similarity_repository.bulk_store_all_similarities(all_similarities, clear_existing=False)

        # Assert
        old = test_db_session.query(ShowSimilarity).filter_by(show_id=99).first()
        assert old is not None

    def test_bulk_store_all_similarities_with_batch_size(
        self, similarity_repository, test_db_session
    ):
        """Test bulk storing with custom batch size."""
        # Arrange
        all_similarities = {
            i: [{"similar_show_id": j, "similarity_score": 0.8} for j in range(10)]
            for i in range(1, 6)
        }

        # Act
        count = similarity_repository.bulk_store_all_similarities(all_similarities, batch_size=10)

        # Assert
        assert count == 50


class TestGetSimilarShows:
    """Tests for get_similar_shows method."""

    def test_get_similar_shows_returns_top_n(
        self, similarity_repository, sample_similarity_records
    ):
        """Test getting top N similar shows."""
        # Act
        results = similarity_repository.get_similar_shows(show_id=1, n=2)

        # Assert
        assert len(results) == 2
        assert all(isinstance(sim, ShowSimilarity) for sim in results)

    def test_get_similar_shows_orders_by_score_descending(
        self, similarity_repository, test_db_session
    ):
        """Test that results are ordered by similarity score descending."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.5),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.9),
            ShowSimilarity(show_id=1, similar_show_id=4, similarity_score=0.7),
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        results = similarity_repository.get_similar_shows(show_id=1, n=3)

        # Assert
        assert results[0].similarity_score == 0.9
        assert results[1].similarity_score == 0.7
        assert results[2].similarity_score == 0.5

    def test_get_similar_shows_filters_by_min_similarity(
        self, similarity_repository, test_db_session
    ):
        """Test filtering by minimum similarity threshold."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.9),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.5),
            ShowSimilarity(show_id=1, similar_show_id=4, similarity_score=0.3),
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        results = similarity_repository.get_similar_shows(show_id=1, n=10, min_similarity=0.6)

        # Assert
        assert len(results) == 1
        assert results[0].similarity_score == 0.9

    def test_get_similar_shows_returns_empty_list_when_none_found(self, similarity_repository):
        """Test getting similar shows when none exist."""
        # Act
        results = similarity_repository.get_similar_shows(show_id=999)

        # Assert
        assert results == []


class TestGetSimilarShowsWithMetadata:
    """Tests for get_similar_shows_with_metadata method."""

    def test_get_similar_shows_with_metadata_returns_complete_info(
        self, similarity_repository, sample_show_metadata_records, test_db_session
    ):
        """Test getting similar shows with their metadata."""
        # Arrange
        similarity = ShowSimilarity(
            show_id=1,
            similar_show_id=2,
            similarity_score=0.85,
            genre_score=0.9,
            text_score=0.8,
            metadata_score=0.85,
        )
        test_db_session.add(similarity)
        test_db_session.commit()

        # Act
        results = similarity_repository.get_similar_shows_with_metadata(show_id=1)

        # Assert
        assert len(results) == 1
        assert results[0]["show_id"] == 2
        assert results[0]["name"] == "Better Call Saul"
        assert results[0]["similarity_score"] == 0.85
        assert results[0]["genre_score"] == 0.9

    def test_get_similar_shows_with_metadata_orders_by_score(
        self, similarity_repository, sample_show_metadata_records, test_db_session
    ):
        """Test that results are ordered by similarity score."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.7),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.9),
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        results = similarity_repository.get_similar_shows_with_metadata(show_id=1, n=2)

        # Assert
        assert len(results) == 2
        assert results[0]["show_id"] == 3  # Higher score
        assert results[1]["show_id"] == 2  # Lower score

    def test_get_similar_shows_with_metadata_filters_by_min_similarity(
        self, similarity_repository, sample_show_metadata_records, test_db_session
    ):
        """Test filtering by minimum similarity with metadata."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.9),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.3),
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        results = similarity_repository.get_similar_shows_with_metadata(
            show_id=1, n=10, min_similarity=0.5
        )

        # Assert
        assert len(results) == 1
        assert results[0]["show_id"] == 2

    def test_get_similar_shows_with_metadata_returns_all_metadata_fields(
        self, similarity_repository, sample_show_metadata_records, test_db_session
    ):
        """Test that all metadata fields are included in results."""
        # Arrange
        similarity = ShowSimilarity(
            show_id=1,
            similar_show_id=2,
            similarity_score=0.85,
            genre_score=0.9,
            text_score=0.8,
            metadata_score=0.85,
        )
        test_db_session.add(similarity)
        test_db_session.commit()

        # Act
        results = similarity_repository.get_similar_shows_with_metadata(show_id=1)

        # Assert
        result = results[0]
        assert "show_id" in result
        assert "name" in result
        assert "genres" in result
        assert "summary" in result
        assert "rating" in result
        assert "type" in result
        assert "language" in result
        assert "network" in result
        assert "similarity_score" in result
        assert "genre_score" in result
        assert "text_score" in result
        assert "metadata_score" in result


class TestGetAllShowIdsWithSimilarities:
    """Tests for get_all_show_ids_with_similarities method."""

    def test_get_all_show_ids_with_similarities_returns_distinct_ids(
        self, similarity_repository, test_db_session
    ):
        """Test getting all show IDs that have similarities."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.8),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.7),
            ShowSimilarity(show_id=2, similar_show_id=1, similarity_score=0.8),
            ShowSimilarity(show_id=3, similar_show_id=1, similarity_score=0.6),
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        result = similarity_repository.get_all_show_ids_with_similarities()

        # Assert
        assert len(result) == 3
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_get_all_show_ids_with_similarities_returns_empty_when_none(
        self, similarity_repository
    ):
        """Test getting show IDs when no similarities exist."""
        # Act
        result = similarity_repository.get_all_show_ids_with_similarities()

        # Assert
        assert result == []


class TestCountSimilarities:
    """Tests for count_similarities method."""

    def test_count_similarities_returns_total_count(self, similarity_repository, test_db_session):
        """Test counting all similarities."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.8),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.7),
            ShowSimilarity(show_id=2, similar_show_id=1, similarity_score=0.8),
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        count = similarity_repository.count_similarities()

        # Assert
        assert count == 3

    def test_count_similarities_for_specific_show(self, similarity_repository, test_db_session):
        """Test counting similarities for a specific show."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.8),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.7),
            ShowSimilarity(show_id=2, similar_show_id=1, similarity_score=0.8),
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        count = similarity_repository.count_similarities(show_id=1)

        # Assert
        assert count == 2

    def test_count_similarities_returns_zero_when_empty(self, similarity_repository):
        """Test counting when no similarities exist."""
        # Act
        count = similarity_repository.count_similarities()

        # Assert
        assert count == 0


class TestDeleteSimilarities:
    """Tests for delete_similarities method."""

    def test_delete_similarities_removes_all_for_show(self, similarity_repository, test_db_session):
        """Test deleting all similarities for a show."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.8),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.7),
            ShowSimilarity(show_id=2, similar_show_id=1, similarity_score=0.8),
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        count = similarity_repository.delete_similarities(show_id=1)

        # Assert
        assert count == 2
        remaining = test_db_session.query(ShowSimilarity).filter_by(show_id=1).all()
        assert len(remaining) == 0
        # Show 2's similarities should still exist
        assert test_db_session.query(ShowSimilarity).filter_by(show_id=2).count() == 1

    def test_delete_similarities_returns_zero_when_none_exist(self, similarity_repository):
        """Test deleting when no similarities exist for show."""
        # Act
        count = similarity_repository.delete_similarities(show_id=999)

        # Assert
        assert count == 0


class TestGetSimilarityStats:
    """Tests for get_similarity_stats method."""

    def test_get_similarity_stats_returns_correct_statistics(
        self, similarity_repository, test_db_session
    ):
        """Test getting similarity statistics."""
        # Arrange
        similarities = [
            ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.8),
            ShowSimilarity(show_id=1, similar_show_id=3, similarity_score=0.7),
            ShowSimilarity(show_id=2, similar_show_id=1, similarity_score=0.8),
        ]
        test_db_session.add_all(similarities)
        test_db_session.commit()

        # Act
        stats = similarity_repository.get_similarity_stats()

        # Assert
        assert stats["total_records"] == 3
        assert stats["unique_shows"] == 2
        assert stats["avg_similarities_per_show"] == 1.5
        assert stats["last_computed"] is not None

    def test_get_similarity_stats_with_empty_database(self, similarity_repository):
        """Test getting stats when database is empty."""
        # Act
        stats = similarity_repository.get_similarity_stats()

        # Assert
        assert stats["total_records"] == 0
        assert stats["unique_shows"] == 0
        assert stats["avg_similarities_per_show"] == 0
        assert stats["last_computed"] is None

    def test_get_similarity_stats_includes_latest_computation_time(
        self, similarity_repository, test_db_session
    ):
        """Test that stats include latest computation time."""
        # Arrange
        similarity = ShowSimilarity(show_id=1, similar_show_id=2, similarity_score=0.8)
        test_db_session.add(similarity)
        test_db_session.commit()

        # Act
        stats = similarity_repository.get_similarity_stats()

        # Assert
        assert stats["last_computed"] is not None
        assert isinstance(stats["last_computed"], datetime)
