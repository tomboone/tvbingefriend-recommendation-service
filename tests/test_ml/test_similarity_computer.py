"""Unit tests for tvbingefriend_recommendation_service.ml.similarity_computer."""

import numpy as np
import pytest

from tvbingefriend_recommendation_service.ml.similarity_computer import SimilarityComputer


class TestSimilarityComputerInit:
    """Tests for SimilarityComputer initialization."""

    def test_init_default_weights(self):
        """Test initialization with default weights."""
        # Act
        computer = SimilarityComputer()

        # Assert
        assert computer.genre_weight == 0.4
        assert computer.text_weight == 0.5
        assert computer.metadata_weight == 0.1

    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        # Act
        computer = SimilarityComputer(genre_weight=0.3, text_weight=0.6, metadata_weight=0.1)

        # Assert
        assert computer.genre_weight == 0.3
        assert computer.text_weight == 0.6
        assert computer.metadata_weight == 0.1


class TestComputeGenreSimilarity:
    """Tests for compute_genre_similarity method."""

    def test_compute_genre_similarity_basic(self, sample_genre_features):
        """Test basic genre similarity computation."""
        # Arrange
        computer = SimilarityComputer()

        # Act
        similarity = computer.compute_genre_similarity(sample_genre_features)

        # Assert
        assert similarity.shape == (3, 3)
        # Diagonal should be 1 (show similar to itself)
        assert np.allclose(np.diag(similarity), 1.0)
        # Similarity should be symmetric
        assert np.allclose(similarity, similarity.T)
        # Similarity should be in [0, 1] with small tolerance for floating point errors
        assert np.all(similarity >= -1e-10)
        assert np.all(similarity <= 1 + 1e-10)

    def test_compute_genre_similarity_identical_shows(self):
        """Test similarity between identical shows."""
        # Arrange
        computer = SimilarityComputer()
        features = np.array([[1, 1, 0], [1, 1, 0]], dtype=float)

        # Act
        similarity = computer.compute_genre_similarity(features)

        # Assert
        # Identical shows should have similarity of 1
        assert pytest.approx(similarity[0, 1], abs=1e-6) == 1.0
        assert pytest.approx(similarity[1, 0], abs=1e-6) == 1.0

    def test_compute_genre_similarity_no_overlap(self):
        """Test similarity between shows with no common genres."""
        # Arrange
        computer = SimilarityComputer()
        features = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

        # Act
        similarity = computer.compute_genre_similarity(features)

        # Assert
        # No overlap should give similarity of 0
        assert pytest.approx(similarity[0, 1], abs=1e-6) == 0.0


class TestComputeTextSimilarity:
    """Tests for compute_text_similarity method."""

    def test_compute_text_similarity_dense(self):
        """Test text similarity with dense matrix."""
        # Arrange
        computer = SimilarityComputer()
        features = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 1.0]])

        # Act
        similarity = computer.compute_text_similarity(features)

        # Assert
        assert similarity.shape == (3, 3)
        assert np.allclose(np.diag(similarity), 1.0)
        assert np.allclose(similarity, similarity.T)

    def test_compute_text_similarity_sparse(self, sample_text_features):
        """Test text similarity with sparse matrix."""
        # Arrange
        computer = SimilarityComputer()

        # Act
        similarity = computer.compute_text_similarity(sample_text_features)

        # Assert
        assert similarity.shape == (3, 3)
        # Allow small tolerance for floating point errors
        assert np.all(similarity >= -1e-10)
        assert np.all(similarity <= 1 + 1e-10)


class TestComputeMetadataSimilarity:
    """Tests for compute_metadata_similarity method."""

    def test_compute_metadata_similarity_basic(
        self, sample_platform_features, sample_type_features, sample_language_features
    ):
        """Test basic metadata similarity computation."""
        # Arrange
        computer = SimilarityComputer()

        # Act
        similarity = computer.compute_metadata_similarity(
            sample_platform_features, sample_type_features, sample_language_features
        )

        # Assert
        assert similarity.shape == (3, 3)
        assert np.allclose(np.diag(similarity), 1.0)
        assert np.allclose(similarity, similarity.T)
        # Allow small tolerance for floating point errors
        assert np.all(similarity >= -1e-10)
        assert np.all(similarity <= 1 + 1e-10)

    def test_compute_metadata_similarity_combines_features(self):
        """Test that metadata similarity combines all features."""
        # Arrange
        computer = SimilarityComputer()
        # Create features where shows differ in different aspects
        platform = np.array([[1, 0], [0, 1], [1, 0]], dtype=float)
        show_type = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
        language = np.array([[1, 0], [1, 0], [1, 0]], dtype=float)

        # Act
        similarity = computer.compute_metadata_similarity(platform, show_type, language)

        # Assert
        # Shows 0 and 2 match on platform and language but not type
        # Shows 0 and 1 match on type and language but not platform
        assert similarity.shape == (3, 3)


class TestComputeHybridSimilarity:
    """Tests for compute_hybrid_similarity method."""

    def test_compute_hybrid_similarity_basic(self, sample_similarity_matrix):
        """Test hybrid similarity computation."""
        # Arrange
        computer = SimilarityComputer(genre_weight=0.4, text_weight=0.5, metadata_weight=0.1)
        genre_sim = sample_similarity_matrix
        text_sim = sample_similarity_matrix * 0.9
        metadata_sim = sample_similarity_matrix * 0.8

        # Act
        hybrid = computer.compute_hybrid_similarity(genre_sim, text_sim, metadata_sim)

        # Assert
        assert hybrid.shape == (3, 3)
        # Check diagonal is close to weighted average of input diagonals
        # genre_sim diagonal is 1.0, text_sim is 0.9, metadata_sim is 0.8
        # hybrid diagonal should be 0.4*1.0 + 0.5*0.9 + 0.1*0.8 = 0.93
        assert np.allclose(np.diag(hybrid), 0.93, atol=1e-6)
        # Allow small tolerance for floating point errors
        assert np.all(hybrid >= -1e-10)
        assert np.all(hybrid <= 1 + 1e-10)

    def test_compute_hybrid_similarity_weights_normalized(self):
        """Test that weights are properly normalized."""
        # Arrange
        computer = SimilarityComputer(genre_weight=2.0, text_weight=3.0, metadata_weight=1.0)
        # Create simple matrices
        genre_sim = np.array([[1.0, 0.6], [0.6, 1.0]])
        text_sim = np.array([[1.0, 0.8], [0.8, 1.0]])
        metadata_sim = np.array([[1.0, 0.4], [0.4, 1.0]])

        # Act
        hybrid = computer.compute_hybrid_similarity(genre_sim, text_sim, metadata_sim)

        # Assert
        # Weights sum to 6, so normalized are: 2/6, 3/6, 1/6
        # hybrid[0,1] = (2/6)*0.6 + (3/6)*0.8 + (1/6)*0.4
        expected = (2 / 6) * 0.6 + (3 / 6) * 0.8 + (1 / 6) * 0.4
        assert pytest.approx(hybrid[0, 1], abs=1e-6) == expected

    def test_compute_hybrid_similarity_equal_weights(self):
        """Test with equal weights for all components."""
        # Arrange
        computer = SimilarityComputer(genre_weight=1.0, text_weight=1.0, metadata_weight=1.0)
        genre_sim = np.array([[1.0, 0.3], [0.3, 1.0]])
        text_sim = np.array([[1.0, 0.6], [0.6, 1.0]])
        metadata_sim = np.array([[1.0, 0.9], [0.9, 1.0]])

        # Act
        hybrid = computer.compute_hybrid_similarity(genre_sim, text_sim, metadata_sim)

        # Assert
        # With equal weights, hybrid should be average
        expected = (0.3 + 0.6 + 0.9) / 3
        assert pytest.approx(hybrid[0, 1], abs=1e-6) == expected


class TestComputeAllSimilarities:
    """Tests for compute_all_similarities method."""

    def test_compute_all_similarities_complete(
        self,
        sample_genre_features,
        sample_text_features,
        sample_platform_features,
        sample_type_features,
        sample_language_features,
    ):
        """Test computing all similarity types."""
        # Arrange
        computer = SimilarityComputer()
        features = {
            "genre_features": sample_genre_features,
            "text_features": sample_text_features,
            "platform_features": sample_platform_features,
            "type_features": sample_type_features,
            "language_features": sample_language_features,
        }

        # Act
        result = computer.compute_all_similarities(features)

        # Assert
        assert "genre_similarity" in result
        assert "text_similarity" in result
        assert "metadata_similarity" in result
        assert "hybrid_similarity" in result

        # All should be same shape
        shape = (3, 3)
        assert result["genre_similarity"].shape == shape
        assert result["text_similarity"].shape == shape
        assert result["metadata_similarity"].shape == shape
        assert result["hybrid_similarity"].shape == shape


class TestGetSimilarityStatistics:
    """Tests for get_similarity_statistics method."""

    def test_get_similarity_statistics_basic(self, sample_similarity_matrix):
        """Test computing statistics for a similarity matrix."""
        # Arrange
        computer = SimilarityComputer()

        # Act
        stats = computer.get_similarity_statistics(sample_similarity_matrix)

        # Assert
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats

        # All stats should be floats
        for _key, value in stats.items():
            assert isinstance(value, float)

        # Logical bounds
        assert 0 <= stats["min"] <= stats["max"] <= 1
        assert 0 <= stats["mean"] <= 1
        assert stats["std"] >= 0

    def test_get_similarity_statistics_excludes_diagonal(self):
        """Test that statistics exclude diagonal (self-similarity)."""
        # Arrange
        computer = SimilarityComputer()
        # Matrix where diagonal is 1 and off-diagonal is 0.5
        matrix = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])

        # Act
        stats = computer.get_similarity_statistics(matrix)

        # Assert
        # Mean should be 0.5 (not affected by diagonal 1.0)
        assert pytest.approx(stats["mean"], abs=1e-6) == 0.5
        # Max should be 0.5 (diagonal excluded)
        assert pytest.approx(stats["max"], abs=1e-6) == 0.5

    def test_get_similarity_statistics_uniform_matrix(self):
        """Test statistics for uniform similarity matrix."""
        # Arrange
        computer = SimilarityComputer()
        matrix = np.full((4, 4), 0.7)
        np.fill_diagonal(matrix, 1.0)

        # Act
        stats = computer.get_similarity_statistics(matrix)

        # Assert
        # All off-diagonal values are 0.7
        assert pytest.approx(stats["mean"], abs=1e-6) == 0.7
        assert pytest.approx(stats["median"], abs=1e-6) == 0.7
        assert pytest.approx(stats["std"], abs=1e-6) == 0.0
        assert pytest.approx(stats["min"], abs=1e-6) == 0.7
        assert pytest.approx(stats["max"], abs=1e-6) == 0.7
