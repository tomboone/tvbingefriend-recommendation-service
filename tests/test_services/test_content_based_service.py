"""Unit tests for ContentBasedRecommendationService."""

from unittest.mock import Mock, patch

import pytest

from tvbingefriend_recommendation_service.services.content_based_service import (
    ContentBasedRecommendationService,
)


class TestContentBasedRecommendationServiceInit:
    """Tests for ContentBasedRecommendationService initialization."""

    def test_init_with_default_values(self):
        """Test initialization with default parameters."""
        # Act
        service = ContentBasedRecommendationService()

        # Assert
        assert service.genre_weight == 0.4
        assert service.text_weight == 0.5
        assert service.metadata_weight == 0.1
        assert service.blob_prefix == "processed"
        assert service.processed_data_dir.name == "processed"

    def test_init_with_custom_weights(self):
        """Test initialization with custom weights."""
        # Act
        service = ContentBasedRecommendationService(
            genre_weight=0.5, text_weight=0.3, metadata_weight=0.2
        )

        # Assert
        assert service.genre_weight == 0.5
        assert service.text_weight == 0.3
        assert service.metadata_weight == 0.2

    def test_init_with_custom_data_dir(self, temp_data_dir):
        """Test initialization with custom data directory."""
        # Act
        service = ContentBasedRecommendationService(processed_data_dir=temp_data_dir)

        # Assert
        assert service.processed_data_dir == temp_data_dir

    def test_init_with_blob_storage_enabled(self, monkeypatch, mock_blob_storage_client):
        """Test initialization with blob storage enabled."""
        # Arrange
        monkeypatch.setenv("USE_BLOB_STORAGE", "true")
        monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "test_connection")
        monkeypatch.setenv("STORAGE_CONTAINER_NAME", "test-container")

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.BlobStorageClient"
        ) as mock_blob_client_class:
            mock_blob_client_class.return_value = mock_blob_storage_client

            # Act
            service = ContentBasedRecommendationService(use_blob=True)

            # Assert
            assert service.use_blob is True
            assert service.blob_client is not None
            mock_blob_client_class.assert_called_once()

    def test_init_with_blob_storage_disabled(self):
        """Test initialization with blob storage disabled."""
        # Act
        service = ContentBasedRecommendationService(use_blob=False)

        # Assert
        assert service.use_blob is False
        assert service.blob_client is None

    def test_init_detects_blob_storage_from_config(self, monkeypatch):
        """Test that blob storage is auto-detected from config."""
        # Arrange
        monkeypatch.setenv("USE_BLOB_STORAGE", "false")

        # Act
        service = ContentBasedRecommendationService()

        # Assert
        assert service.use_blob is False


class TestGetDataDir:
    """Tests for _get_data_dir method."""

    def test_get_data_dir_returns_local_dir_when_blob_disabled(self, temp_data_dir):
        """Test that local directory is returned when blob is disabled."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir, use_blob=False
        )

        # Act
        data_dir = service._get_data_dir()

        # Assert
        assert data_dir == temp_data_dir

    def test_get_data_dir_downloads_from_blob_when_enabled(self, mock_blob_storage_client):
        """Test that data is downloaded from blob when enabled."""
        # Arrange
        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.BlobStorageClient"
        ) as mock_blob_class:
            mock_blob_class.return_value = mock_blob_storage_client
            service = ContentBasedRecommendationService(use_blob=True)

            # Act
            data_dir = service._get_data_dir()

            # Assert
            assert data_dir is not None
            assert data_dir == service._temp_dir
            mock_blob_storage_client.download_directory.assert_called_once()

    def test_get_data_dir_reuses_temp_dir_on_subsequent_calls(self, mock_blob_storage_client):
        """Test that temp directory is reused on subsequent calls."""
        # Arrange
        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.BlobStorageClient"
        ) as mock_blob_class:
            mock_blob_class.return_value = mock_blob_storage_client
            service = ContentBasedRecommendationService(use_blob=True)

            # Act
            data_dir1 = service._get_data_dir()
            data_dir2 = service._get_data_dir()

            # Assert
            assert data_dir1 == data_dir2
            # download_directory should only be called once
            assert mock_blob_storage_client.download_directory.call_count == 1

    def test_get_data_dir_raises_error_when_blob_client_not_initialized(self):
        """Test error when blob client is not initialized."""
        # Arrange
        service = ContentBasedRecommendationService(use_blob=False)
        service.use_blob = True  # Force blob mode without client

        # Act & Assert
        with pytest.raises(RuntimeError, match="Blob client not initialized"):
            service._get_data_dir()


class TestLoadSimilarityMatrices:
    """Tests for _load_similarity_matrices method."""

    def test_load_similarity_matrices_loads_all_matrices(self, temp_data_dir_with_files):
        """Test loading all similarity matrices from disk."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        # Act
        service._load_similarity_matrices()

        # Assert
        assert service._genre_similarity is not None
        assert service._text_similarity is not None
        assert service._metadata_similarity is not None
        assert service._hybrid_similarity is not None

    def test_load_similarity_matrices_computes_hybrid_similarity(self, temp_data_dir_with_files):
        """Test that hybrid similarity is computed from individual matrices."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files,
            use_blob=False,
            genre_weight=0.4,
            text_weight=0.5,
            metadata_weight=0.1,
        )

        # Act
        service._load_similarity_matrices()

        # Assert
        assert service._hybrid_similarity is not None
        # Verify shape matches
        assert service._hybrid_similarity.shape == service._genre_similarity.shape

    def test_load_similarity_matrices_is_idempotent(self, temp_data_dir_with_files):
        """Test that calling load multiple times doesn't reload data."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        # Act
        service._load_similarity_matrices()
        first_hybrid = service._hybrid_similarity
        service._load_similarity_matrices()
        second_hybrid = service._hybrid_similarity

        # Assert
        assert first_hybrid is second_hybrid  # Same object reference


class TestLoadShowMappings:
    """Tests for _load_show_mappings method."""

    def test_load_show_mappings_creates_bidirectional_mappings(self, temp_data_dir_with_files):
        """Test that show ID to index mappings are created correctly."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        # Act
        service._load_show_mappings()

        # Assert
        assert service._show_id_to_index is not None
        assert service._index_to_show_id is not None
        assert len(service._show_id_to_index) == len(service._index_to_show_id)

    def test_load_show_mappings_is_idempotent(self, temp_data_dir_with_files):
        """Test that calling load multiple times doesn't reload data."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        # Act
        service._load_show_mappings()
        first_mapping = service._show_id_to_index
        service._load_show_mappings()
        second_mapping = service._show_id_to_index

        # Assert
        assert first_mapping is second_mapping


class TestGetRecommendationsFromMatrix:
    """Tests for get_recommendations_from_matrix method."""

    def test_get_recommendations_from_matrix_returns_top_n_shows(self, temp_data_dir_with_files):
        """Test getting top N recommendations from similarity matrix."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        # Act
        recommendations = service.get_recommendations_from_matrix(show_id=1, n=2)

        # Assert
        assert len(recommendations) <= 2
        assert all("show_id" in rec for rec in recommendations)
        assert all("similarity_score" in rec for rec in recommendations)

    def test_get_recommendations_from_matrix_excludes_self(self, temp_data_dir_with_files):
        """Test that recommendations exclude the source show itself."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        # Act
        recommendations = service.get_recommendations_from_matrix(show_id=1, n=10)

        # Assert
        # Should not include the source show
        assert all(rec["show_id"] != 1 for rec in recommendations)

    def test_get_recommendations_from_matrix_filters_by_min_similarity(
        self, temp_data_dir_with_files
    ):
        """Test filtering recommendations by minimum similarity."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        # Act
        recommendations = service.get_recommendations_from_matrix(
            show_id=1, n=10, min_similarity=0.9
        )

        # Assert
        # All recommendations should have similarity >= 0.9
        assert all(rec["similarity_score"] >= 0.9 for rec in recommendations)

    def test_get_recommendations_from_matrix_returns_empty_for_unknown_show(
        self, temp_data_dir_with_files
    ):
        """Test that empty list is returned for unknown show ID."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        # Act
        recommendations = service.get_recommendations_from_matrix(show_id=9999, n=10)

        # Assert
        assert recommendations == []

    def test_get_recommendations_from_matrix_includes_component_scores(
        self, temp_data_dir_with_files
    ):
        """Test that recommendations include all component scores."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        # Act
        recommendations = service.get_recommendations_from_matrix(show_id=1, n=1)

        # Assert
        if recommendations:
            rec = recommendations[0]
            assert "genre_score" in rec
            assert "text_score" in rec
            assert "metadata_score" in rec
            assert isinstance(rec["genre_score"], float)
            assert isinstance(rec["text_score"], float)
            assert isinstance(rec["metadata_score"], float)

    def test_get_recommendations_from_matrix_orders_by_similarity_descending(
        self, temp_data_dir_with_files
    ):
        """Test that recommendations are ordered by similarity score (descending)."""
        # Arrange
        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        # Act
        recommendations = service.get_recommendations_from_matrix(show_id=1, n=3)

        # Assert
        if len(recommendations) >= 2:
            scores = [rec["similarity_score"] for rec in recommendations]
            assert scores == sorted(scores, reverse=True)


class TestGetRecommendationsFromDb:
    """Tests for get_recommendations_from_db method."""

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_get_recommendations_from_db_queries_database(self, mock_session_local):
        """Test getting recommendations from database."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_repo = Mock()
        mock_repo.get_similar_shows_with_metadata.return_value = [
            {"show_id": 2, "name": "Similar Show", "similarity_score": 0.85}
        ]

        service = ContentBasedRecommendationService(use_blob=False)

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.SimilarityRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            # Act
            recommendations = service.get_recommendations_from_db(show_id=1, n=10)

            # Assert
            assert len(recommendations) == 1
            assert recommendations[0]["show_id"] == 2
            mock_repo.get_similar_shows_with_metadata.assert_called_once_with(
                show_id=1, n=10, min_similarity=0.0
            )

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_get_recommendations_from_db_closes_session(self, mock_session_local):
        """Test that database session is properly closed."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        service = ContentBasedRecommendationService(use_blob=False)

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.SimilarityRepository"
        ):
            # Act
            service.get_recommendations_from_db(show_id=1)

            # Assert
            mock_db.close.assert_called_once()

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_get_recommendations_from_db_with_min_similarity(self, mock_session_local):
        """Test getting recommendations with minimum similarity filter."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_repo = Mock()
        mock_repo.get_similar_shows_with_metadata.return_value = []

        service = ContentBasedRecommendationService(use_blob=False)

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.SimilarityRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            # Act
            service.get_recommendations_from_db(show_id=1, min_similarity=0.5)

            # Assert
            mock_repo.get_similar_shows_with_metadata.assert_called_once_with(
                show_id=1, n=10, min_similarity=0.5
            )


class TestComputeAndStoreAllSimilarities:
    """Tests for compute_and_store_all_similarities method."""

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_compute_and_store_all_similarities_processes_all_shows(
        self, mock_session_local, temp_data_dir_with_files
    ):
        """Test computing and storing similarities for all shows."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_repo = Mock()
        mock_repo.bulk_store_all_similarities.return_value = 6
        mock_repo.get_similarity_stats.return_value = {
            "total_records": 6,
            "unique_shows": 3,
            "avg_similarities_per_show": 2.0,
        }

        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.SimilarityRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            # Act
            stats = service.compute_and_store_all_similarities(top_n_per_show=2)

            # Assert
            assert stats["computed_shows"] == 3
            assert stats["top_n_per_show"] == 2
            mock_repo.bulk_store_all_similarities.assert_called_once()

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_compute_and_store_all_similarities_uses_min_similarity_threshold(
        self, mock_session_local, temp_data_dir_with_files
    ):
        """Test that minimum similarity threshold is applied."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_repo = Mock()
        mock_repo.bulk_store_all_similarities.return_value = 3
        mock_repo.get_similarity_stats.return_value = {
            "total_records": 3,
            "unique_shows": 3,
            "avg_similarities_per_show": 1.0,
        }

        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.SimilarityRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            # Act
            stats = service.compute_and_store_all_similarities(top_n_per_show=5, min_similarity=0.5)

            # Assert
            assert stats["min_similarity"] == 0.5

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_compute_and_store_all_similarities_closes_session(
        self, mock_session_local, temp_data_dir_with_files
    ):
        """Test that database session is properly closed."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_repo = Mock()
        mock_repo.bulk_store_all_similarities.return_value = 0
        mock_repo.get_similarity_stats.return_value = {
            "total_records": 0,
            "unique_shows": 0,
            "avg_similarities_per_show": 0,
        }

        service = ContentBasedRecommendationService(
            processed_data_dir=temp_data_dir_with_files, use_blob=False
        )

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.SimilarityRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            # Act
            service.compute_and_store_all_similarities()

            # Assert
            mock_db.close.assert_called_once()


class TestSyncMetadataToDb:
    """Tests for sync_metadata_to_db method."""

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_sync_metadata_to_db_stores_shows(self, mock_session_local):
        """Test syncing show metadata to database."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_repo = Mock()
        mock_repo.bulk_store_shows.return_value = 2

        shows_data = [
            {
                "id": 1,
                "name": "Show 1",
                "genres": ["Drama"],
                "summary": "Summary 1",
                "rating_avg": 8.5,
                "type": "Scripted",
                "language": "English",
            },
            {
                "id": 2,
                "name": "Show 2",
                "genres": ["Comedy"],
                "summary": "Summary 2",
                "rating_avg": 7.5,
                "type": "Scripted",
                "language": "English",
            },
        ]

        service = ContentBasedRecommendationService(use_blob=False)

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.MetadataRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            # Act
            count = service.sync_metadata_to_db(shows_data)

            # Assert
            assert count == 2
            mock_repo.bulk_store_shows.assert_called_once()

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_sync_metadata_to_db_handles_nan_ratings(self, mock_session_local):
        """Test that NaN ratings are converted to None."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_repo = Mock()
        mock_repo.bulk_store_shows.return_value = 1

        shows_data = [
            {
                "id": 1,
                "name": "Show 1",
                "genres": ["Drama"],
                "summary": "Summary 1",
                "rating_avg": float("nan"),  # NaN value
                "type": "Scripted",
                "language": "English",
            }
        ]

        service = ContentBasedRecommendationService(use_blob=False)

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.MetadataRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            # Act
            service.sync_metadata_to_db(shows_data)

            # Assert
            # Check that the rating was converted to None
            call_args = mock_repo.bulk_store_shows.call_args[0][0]
            assert call_args[0]["rating"] is None

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_sync_metadata_to_db_uses_summary_clean_if_available(self, mock_session_local):
        """Test that summary_clean is used when available."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_repo = Mock()
        mock_repo.bulk_store_shows.return_value = 1

        shows_data = [
            {
                "id": 1,
                "name": "Show 1",
                "genres": ["Drama"],
                "summary": "<p>HTML Summary</p>",
                "summary_clean": "Clean Summary",
                "rating_avg": 8.5,
                "type": "Scripted",
                "language": "English",
            }
        ]

        service = ContentBasedRecommendationService(use_blob=False)

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.MetadataRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            # Act
            service.sync_metadata_to_db(shows_data)

            # Assert
            call_args = mock_repo.bulk_store_shows.call_args[0][0]
            assert call_args[0]["summary"] == "Clean Summary"

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_sync_metadata_to_db_extracts_platform_from_show_data(self, mock_session_local):
        """Test that platform is correctly extracted from show data."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_repo = Mock()
        mock_repo.bulk_store_shows.return_value = 1

        shows_data = [
            {
                "id": 1,
                "name": "Show 1",
                "genres": ["Drama"],
                "summary": "Summary",
                "platform": "Netflix",
                "rating_avg": 8.5,
                "type": "Scripted",
                "language": "English",
            }
        ]

        service = ContentBasedRecommendationService(use_blob=False)

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.MetadataRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            # Act
            service.sync_metadata_to_db(shows_data)

            # Assert
            call_args = mock_repo.bulk_store_shows.call_args[0][0]
            assert call_args[0]["network"] == "Netflix"

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_sync_metadata_to_db_closes_session(self, mock_session_local):
        """Test that database session is properly closed."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_repo = Mock()
        mock_repo.bulk_store_shows.return_value = 0

        service = ContentBasedRecommendationService(use_blob=False)

        with patch(
            "tvbingefriend_recommendation_service.services.content_based_service.MetadataRepository"
        ) as mock_repo_class:
            mock_repo_class.return_value = mock_repo

            # Act
            service.sync_metadata_to_db([])

            # Assert
            mock_db.close.assert_called_once()


class TestGetStats:
    """Tests for get_stats method."""

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_get_stats_returns_all_statistics(self, mock_session_local):
        """Test getting comprehensive statistics."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_similarity_repo = Mock()
        mock_similarity_repo.get_similarity_stats.return_value = {
            "total_records": 100,
            "unique_shows": 50,
            "avg_similarities_per_show": 2.0,
        }

        mock_metadata_repo = Mock()
        mock_metadata_repo.count_shows.return_value = 50

        service = ContentBasedRecommendationService(
            use_blob=False, genre_weight=0.4, text_weight=0.5, metadata_weight=0.1
        )

        with (
            patch(
                "tvbingefriend_recommendation_service.services.content_based_service.SimilarityRepository"
            ) as mock_sim_class,
            patch(
                "tvbingefriend_recommendation_service.services.content_based_service.MetadataRepository"
            ) as mock_meta_class,
        ):
            mock_sim_class.return_value = mock_similarity_repo
            mock_meta_class.return_value = mock_metadata_repo

            # Act
            stats = service.get_stats()

            # Assert
            assert "similarity_stats" in stats
            assert "cached_shows" in stats
            assert "weights" in stats
            assert stats["cached_shows"] == 50
            assert stats["weights"]["genre"] == 0.4
            assert stats["weights"]["text"] == 0.5
            assert stats["weights"]["metadata"] == 0.1

    @patch("tvbingefriend_recommendation_service.services.content_based_service.SessionLocal")
    def test_get_stats_closes_session(self, mock_session_local):
        """Test that database session is properly closed."""
        # Arrange
        mock_db = Mock()
        mock_session_local.return_value = mock_db

        mock_similarity_repo = Mock()
        mock_similarity_repo.get_similarity_stats.return_value = {}

        mock_metadata_repo = Mock()
        mock_metadata_repo.count_shows.return_value = 0

        service = ContentBasedRecommendationService(use_blob=False)

        with (
            patch(
                "tvbingefriend_recommendation_service.services.content_based_service.SimilarityRepository"
            ) as mock_sim_class,
            patch(
                "tvbingefriend_recommendation_service.services.content_based_service.MetadataRepository"
            ) as mock_meta_class,
        ):
            mock_sim_class.return_value = mock_similarity_repo
            mock_meta_class.return_value = mock_metadata_repo

            # Act
            service.get_stats()

            # Assert
            mock_db.close.assert_called_once()
