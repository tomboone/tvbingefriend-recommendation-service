"""Unit tests for tvbingefriend_recommendation_service.ml.feature_extractor."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from scipy.sparse import csr_matrix

from tvbingefriend_recommendation_service.ml.feature_extractor import FeatureExtractor


class TestFeatureExtractorInit:
    """Tests for FeatureExtractor initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        # Act
        extractor = FeatureExtractor()

        # Assert
        assert extractor.max_text_features == 500
        assert extractor.text_min_df == 2
        assert extractor.text_max_df == 0.8
        assert extractor.top_n_platforms == 20
        assert extractor.top_n_languages == 5
        assert extractor.genre_encoder is None
        assert extractor.tfidf_vectorizer is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        # Act
        extractor = FeatureExtractor(
            max_text_features=1000,
            text_min_df=3,
            text_max_df=0.9,
            top_n_platforms=30,
            top_n_languages=10
        )

        # Assert
        assert extractor.max_text_features == 1000
        assert extractor.text_min_df == 3
        assert extractor.text_max_df == 0.9
        assert extractor.top_n_platforms == 30
        assert extractor.top_n_languages == 10


class TestFitTransformGenreFeatures:
    """Tests for fit_transform_genre_features method."""

    def test_fit_transform_genre_features_basic(self):
        """Test basic genre feature extraction."""
        # Arrange
        extractor = FeatureExtractor()
        genres_list = [
            ['Drama', 'Crime'],
            ['Comedy', 'Drama'],
            ['Action', 'Thriller']
        ]

        # Act
        features, encoder = extractor.fit_transform_genre_features(genres_list)

        # Assert
        assert features.shape[0] == 3
        assert features.shape[1] == 5  # Number of unique genres
        assert encoder == extractor.genre_encoder
        assert len(encoder.classes_) == 5

    def test_fit_transform_genre_features_empty_genres(self):
        """Test with some empty genre lists."""
        # Arrange
        extractor = FeatureExtractor()
        genres_list = [
            ['Drama'],
            [],
            ['Comedy']
        ]

        # Act
        features, encoder = extractor.fit_transform_genre_features(genres_list)

        # Assert
        assert features.shape[0] == 3
        assert np.sum(features[1]) == 0  # Second show has no genres


class TestFitTransformTextFeatures:
    """Tests for fit_transform_text_features method."""

    def test_fit_transform_text_features_basic(self):
        """Test basic text feature extraction."""
        # Arrange
        # Use min_df=1 for small test data to avoid "no terms remain" error
        extractor = FeatureExtractor(max_text_features=10, text_min_df=1)
        summaries = [
            "A chemistry teacher cooks meth",
            "A lawyer defends criminals",
            "Office workers in a paper company"
        ]

        # Act
        features, vectorizer = extractor.fit_transform_text_features(summaries)

        # Assert
        assert isinstance(features, csr_matrix)
        assert features.shape[0] == 3
        assert features.shape[1] <= 10
        assert vectorizer == extractor.tfidf_vectorizer
        assert len(vectorizer.get_feature_names_out()) > 0

    def test_fit_transform_text_features_with_html(self):
        """Test text feature extraction with HTML content."""
        # Arrange
        # Use min_df=1 for small test data to avoid "no terms remain" error
        extractor = FeatureExtractor(text_min_df=1)
        summaries = [
            "<p>A chemistry teacher</p>",
            "<b>A lawyer</b>",
            "Office workers"
        ]

        # Act
        features, vectorizer = extractor.fit_transform_text_features(summaries)

        # Assert
        assert features.shape[0] == 3
        assert features.shape[1] > 0


class TestFitTransformPlatformFeatures:
    """Tests for fit_transform_platform_features method."""

    def test_fit_transform_platform_features_basic(self):
        """Test basic platform feature extraction."""
        # Arrange
        extractor = FeatureExtractor(top_n_platforms=2)
        platforms = ['Netflix', 'HBO', 'Netflix', 'AMC', 'HBO']

        # Act
        features = extractor.fit_transform_platform_features(platforms)

        # Assert
        assert features.shape == (5, 3)  # 5 shows, 2 top platforms + 1 other
        assert extractor.top_platforms is not None
        assert len(extractor.top_platforms) == 2

    def test_fit_transform_platform_features_other_category(self):
        """Test that less common platforms go to 'other' category."""
        # Arrange
        extractor = FeatureExtractor(top_n_platforms=1)
        platforms = ['Netflix', 'Netflix', 'HBO', 'AMC']

        # Act
        features = extractor.fit_transform_platform_features(platforms)

        # Assert
        # Netflix should be top platform, HBO and AMC should be in "other"
        assert features[0, 0] == 1  # Netflix
        assert features[2, -1] == 1  # HBO -> other
        assert features[3, -1] == 1  # AMC -> other

    def test_fit_transform_platform_features_with_none(self):
        """Test platform features with None values."""
        # Arrange
        extractor = FeatureExtractor(top_n_platforms=2)
        platforms = ['Netflix', None, 'HBO', 'Netflix']

        # Act
        features = extractor.fit_transform_platform_features(platforms)

        # Assert
        assert features.shape[0] == 4
        # None platform should end up in a category


class TestFitTransformTypeFeatures:
    """Tests for fit_transform_type_features method."""

    def test_fit_transform_type_features_basic(self):
        """Test basic type feature extraction."""
        # Arrange
        extractor = FeatureExtractor()
        types = ['Scripted', 'Reality', 'Scripted', 'Documentary']

        # Act
        features = extractor.fit_transform_type_features(types)

        # Assert
        assert features.shape[0] == 4
        assert features.shape[1] == 3  # 3 unique types
        assert np.sum(features[0]) == 1  # One-hot encoded

    def test_fit_transform_type_features_single_type(self):
        """Test with all shows of same type."""
        # Arrange
        extractor = FeatureExtractor()
        types = ['Scripted', 'Scripted', 'Scripted']

        # Act
        features = extractor.fit_transform_type_features(types)

        # Assert
        assert features.shape[0] == 3
        assert features.shape[1] == 1  # Only one type
        assert np.all(features == 1)


class TestFitTransformLanguageFeatures:
    """Tests for fit_transform_language_features method."""

    def test_fit_transform_language_features_basic(self):
        """Test basic language feature extraction."""
        # Arrange
        extractor = FeatureExtractor(top_n_languages=2)
        languages = ['English', 'Spanish', 'English', 'French', 'English']

        # Act
        features = extractor.fit_transform_language_features(languages)

        # Assert
        assert features.shape == (5, 3)  # 5 shows, 2 top languages + 1 other
        assert extractor.top_languages is not None
        assert len(extractor.top_languages) == 2

    def test_fit_transform_language_features_other_category(self):
        """Test that less common languages go to 'other' category."""
        # Arrange
        extractor = FeatureExtractor(top_n_languages=1)
        languages = ['English', 'English', 'Spanish', 'French']

        # Act
        features = extractor.fit_transform_language_features(languages)

        # Assert
        # English should be top, Spanish and French should be "other"
        assert features[0, 0] == 1  # English
        assert features[2, -1] == 1  # Spanish -> other
        assert features[3, -1] == 1  # French -> other


class TestExtractAllFeatures:
    """Tests for extract_all_features method."""

    def test_extract_all_features_complete(self, sample_shows_df):
        """Test extracting all features from DataFrame."""
        # Arrange
        # Use min_df=1 for small test data to avoid "no terms remain" error
        extractor = FeatureExtractor(max_text_features=10, text_min_df=1)

        # Act
        result = extractor.extract_all_features(sample_shows_df)

        # Assert
        assert 'genre_features' in result
        assert 'text_features' in result
        assert 'platform_features' in result
        assert 'type_features' in result
        assert 'language_features' in result
        assert 'genre_encoder' in result
        assert 'tfidf_vectorizer' in result

        # Check shapes
        n_shows = len(sample_shows_df)
        assert result['genre_features'].shape[0] == n_shows
        assert result['text_features'].shape[0] == n_shows
        assert result['platform_features'].shape[0] == n_shows
        assert result['type_features'].shape[0] == n_shows
        assert result['language_features'].shape[0] == n_shows

    def test_extract_all_features_encoders_fitted(self, sample_shows_df):
        """Test that encoders are properly fitted."""
        # Arrange
        # Use min_df=1 for small test data to avoid "no terms remain" error
        extractor = FeatureExtractor(text_min_df=1)

        # Act
        result = extractor.extract_all_features(sample_shows_df)

        # Assert
        assert extractor.genre_encoder is not None
        assert extractor.tfidf_vectorizer is not None
        assert extractor.top_platforms is not None
        assert extractor.top_languages is not None

    def test_extract_all_features_with_missing_values(self):
        """Test extraction with missing values in DataFrame."""
        # Arrange
        # Use min_df=1 for small test data to avoid "no terms remain" error
        extractor = FeatureExtractor(text_min_df=1)
        df = pd.DataFrame([
            {
                'id': 1,
                'name': 'Show 1',
                'genres': ['Drama'],
                'summary_clean': 'Summary 1',
                'type': 'Scripted',
                'language': 'English',
                'platform': None,
                'rating_avg': 8.5
            },
            {
                'id': 2,
                'name': 'Show 2',
                'genres': ['Comedy'],
                'summary_clean': None,
                'type': 'Scripted',
                'language': 'English',
                'platform': 'Netflix',
                'rating_avg': 7.5
            }
        ])

        # Act
        result = extractor.extract_all_features(df)

        # Assert
        assert result['genre_features'].shape[0] == 2
        assert result['text_features'].shape[0] == 2
