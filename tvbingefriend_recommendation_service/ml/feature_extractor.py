"""Feature extraction for TV show recommendations."""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.preprocessing import MultiLabelBinarizer  # type: ignore
from scipy.sparse import csr_matrix
import logging

from tvbingefriend_recommendation_service.ml.text_processor import clean_html

logger = logging.getLogger(__name__)


# noinspection PyMethodMayBeStatic
class FeatureExtractor:
    """Extract features from TV show data for content-based recommendations."""

    def __init__(
        self,
        max_text_features: int = 500,
        text_min_df: int = 2,
        text_max_df: float = 0.8,
        top_n_platforms: int = 20,
        top_n_languages: int = 5
    ):
        """
        Initialize feature extractor.

        Args:
            max_text_features: Maximum number of TF-IDF features
            text_min_df: Minimum document frequency for TF-IDF
            text_max_df: Maximum document frequency for TF-IDF
            top_n_platforms: Number of top platforms to encode
            top_n_languages: Number of top languages to encode
        """
        self.max_text_features = max_text_features
        self.text_min_df = text_min_df
        self.text_max_df = text_max_df
        self.top_n_platforms = top_n_platforms
        self.top_n_languages = top_n_languages

        # Encoders (fitted during transform)
        self.genre_encoder: Optional[MultiLabelBinarizer] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.top_platforms: Optional[List[str]] = None
        self.top_languages: Optional[List[str]] = None

    def fit_transform_genre_features(
        self,
        genres_list: List[List[str]]
    ) -> Tuple[np.ndarray, MultiLabelBinarizer]:
        """
        Extract genre features using multi-hot encoding.

        Args:
            genres_list: List of genre lists for each show

        Returns:
            (genre_features, encoder) tuple
        """
        logger.info("Extracting genre features...")

        self.genre_encoder = MultiLabelBinarizer()
        genre_features = self.genre_encoder.fit_transform(genres_list)

        logger.info(f" Genre features: {genre_features.shape}")
        logger.info(f"  Unique genres: {len(self.genre_encoder.classes_)}")

        return genre_features, self.genre_encoder

    def fit_transform_text_features(
        self,
        summaries: List[str]
    ) -> Tuple[csr_matrix, TfidfVectorizer]:
        """
        Extract text features using TF-IDF on summaries.

        Args:
            summaries: List of cleaned summaries

        Returns:
            (text_features, vectorizer) tuple
        """
        logger.info("Extracting text features (TF-IDF)...")

        # Clean summaries
        cleaned_summaries = [clean_html(s) for s in summaries]

        # Initialize and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_text_features,
            min_df=self.text_min_df,
            max_df=self.text_max_df,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            strip_accents='unicode'
        )

        text_features = self.tfidf_vectorizer.fit_transform(cleaned_summaries)

        logger.info(f" Text features: {text_features.shape}")
        logger.info(f"  Vocabulary size: {len(self.tfidf_vectorizer.get_feature_names_out())}")
        logger.info(
            f"  Sparsity: {(1 - text_features.nnz / (text_features.shape[0] * text_features.shape[1])) * 100:.2f}%"
        )

        return text_features, self.tfidf_vectorizer

    def fit_transform_platform_features(
        self,
        platforms: List[Optional[str]]
    ) -> np.ndarray:
        """
        Extract platform features using one-hot encoding (top N platforms).

        Args:
            platforms: List of platform names

        Returns:
            platform_features array
        """
        logger.info("Extracting platform features...")

        # Convert to pandas Series for easy counting
        platform_series = pd.Series(platforms)

        # Get top N platforms
        platform_counts = platform_series.value_counts()
        self.top_platforms = platform_counts.head(self.top_n_platforms).index.tolist()

        # Create binary features
        platform_features = np.zeros((len(platforms), self.top_n_platforms + 1))

        for i, platform in enumerate(platforms):
            if platform in self.top_platforms:
                idx = self.top_platforms.index(platform)
                platform_features[i, idx] = 1
            else:
                # "Other" category
                platform_features[i, -1] = 1

        logger.info(f" Platform features: {platform_features.shape}")
        logger.info(f"  Top platforms: {self.top_platforms[:5]}...")

        return platform_features

    def fit_transform_type_features(
        self,
        types: List[str]
    ) -> np.ndarray:
        """
        Extract type features using one-hot encoding.

        Args:
            types: List of show types

        Returns:
            type_features array
        """
        logger.info("Extracting type features...")

        # Use pandas get_dummies for simple one-hot encoding
        type_df = pd.get_dummies(pd.Series(types), prefix='type')
        type_features = type_df.values

        logger.info(f" Type features: {type_features.shape}")
        logger.info(f"  Types: {list(type_df.columns)}")

        return type_features

    def fit_transform_language_features(
        self,
        languages: List[str]
    ) -> np.ndarray:
        """
        Extract language features using one-hot encoding (top N languages).

        Args:
            languages: List of languages

        Returns:
            language_features array
        """
        logger.info("Extracting language features...")

        # Convert to pandas Series
        language_series = pd.Series(languages)

        # Get top N languages
        language_counts = language_series.value_counts()
        self.top_languages = language_counts.head(self.top_n_languages).index.tolist()

        # Create binary features
        language_features = np.zeros((len(languages), self.top_n_languages + 1))

        for i, language in enumerate(languages):
            if language in self.top_languages:
                idx = self.top_languages.index(language)
                language_features[i, idx] = 1
            else:
                # "Other" category
                language_features[i, -1] = 1

        logger.info(f" Language features: {language_features.shape}")
        logger.info(f"  Top languages: {self.top_languages}")

        return language_features

    def extract_all_features(
        self,
        shows_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Extract all features from shows DataFrame.

        Args:
            shows_df: DataFrame with columns: id, name, genres, summary_clean,
                     type, language, platform, rating_avg

        Returns:
            Dictionary with feature arrays and encoders
        """
        logger.info("="*60)
        logger.info("EXTRACTING ALL FEATURES")
        logger.info("="*60)
        logger.info(f"Processing {len(shows_df)} shows...")

        # Extract each feature group
        genre_features, genre_encoder = self.fit_transform_genre_features(
            shows_df['genres'].tolist()
        )

        text_features, tfidf_vectorizer = self.fit_transform_text_features(
            shows_df['summary_clean'].tolist()
        )

        platform_features = self.fit_transform_platform_features(
            shows_df['platform'].tolist()
        )

        type_features = self.fit_transform_type_features(
            shows_df['type'].tolist()
        )

        language_features = self.fit_transform_language_features(
            shows_df['language'].tolist()
        )

        logger.info("="*60)
        logger.info("FEATURE EXTRACTION COMPLETE")
        logger.info("="*60)

        total_features = (
            genre_features.shape[1] +
            text_features.shape[1] +
            platform_features.shape[1] +
            type_features.shape[1] +
            language_features.shape[1]
        )
        logger.info(f"Total feature dimensions: {total_features}")

        return {
            'genre_features': genre_features,
            'text_features': text_features,
            'platform_features': platform_features,
            'type_features': type_features,
            'language_features': language_features,
            'genre_encoder': genre_encoder,
            'tfidf_vectorizer': tfidf_vectorizer
        }
