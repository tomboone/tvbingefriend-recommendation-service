"""Compute similarity matrices for TV show recommendations."""
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse
import logging

logger = logging.getLogger(__name__)


class SimilarityComputer:
    """Compute similarity matrices from feature arrays."""

    def __init__(
        self,
        genre_weight: float = 0.4,
        text_weight: float = 0.5,
        metadata_weight: float = 0.1
    ):
        """
        Initialize similarity computer.

        Args:
            genre_weight: Weight for genre similarity
            text_weight: Weight for text/content similarity
            metadata_weight: Weight for metadata (platform, type, language) similarity
        """
        self.genre_weight = genre_weight
        self.text_weight = text_weight
        self.metadata_weight = metadata_weight

    def compute_genre_similarity(self, genre_features: np.ndarray) -> np.ndarray:
        """
        Compute genre similarity using cosine similarity.

        Args:
            genre_features: Binary genre feature matrix (n_shows x n_genres)

        Returns:
            Genre similarity matrix (n_shows x n_shows)
        """
        logger.info("Computing genre similarity...")
        similarity = cosine_similarity(genre_features)
        logger.info(f" Genre similarity: {similarity.shape}, range [{similarity.min():.3f}, {similarity.max():.3f}]")
        return similarity

    def compute_text_similarity(self, text_features) -> np.ndarray:
        """
        Compute text similarity using cosine similarity on TF-IDF vectors.

        Args:
            text_features: TF-IDF feature matrix (can be sparse)

        Returns:
            Text similarity matrix (n_shows x n_shows)
        """
        logger.info("Computing text similarity...")
        similarity = cosine_similarity(text_features)
        logger.info(f" Text similarity: {similarity.shape}, range [{similarity.min():.3f}, {similarity.max():.3f}]")
        return similarity

    def compute_metadata_similarity(
        self,
        platform_features: np.ndarray,
        type_features: np.ndarray,
        language_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute metadata similarity by combining platform, type, and language features.

        Args:
            platform_features: Platform feature matrix
            type_features: Type feature matrix
            language_features: Language feature matrix

        Returns:
            Metadata similarity matrix (n_shows x n_shows)
        """
        logger.info("Computing metadata similarity...")

        # Combine metadata features
        metadata_features = np.hstack([
            platform_features,
            type_features,
            language_features
        ])

        similarity = cosine_similarity(metadata_features)
        logger.info(f" Metadata similarity: {similarity.shape}, range [{similarity.min():.3f}, {similarity.max():.3f}]")
        return similarity

    def compute_hybrid_similarity(
        self,
        genre_similarity: np.ndarray,
        text_similarity: np.ndarray,
        metadata_similarity: np.ndarray
    ) -> np.ndarray:
        """
        Compute hybrid similarity as weighted combination.

        Args:
            genre_similarity: Genre similarity matrix
            text_similarity: Text similarity matrix
            metadata_similarity: Metadata similarity matrix

        Returns:
            Hybrid similarity matrix (n_shows x n_shows)
        """
        logger.info("Computing hybrid similarity...")

        # Normalize weights
        total_weight = self.genre_weight + self.text_weight + self.metadata_weight
        genre_w = self.genre_weight / total_weight
        text_w = self.text_weight / total_weight
        metadata_w = self.metadata_weight / total_weight

        logger.info(f"  Weights - Genre: {genre_w:.2f}, Text: {text_w:.2f}, Metadata: {metadata_w:.2f}")

        # Weighted combination
        hybrid_similarity = (
            genre_w * genre_similarity +
            text_w * text_similarity +
            metadata_w * metadata_similarity
        )

        logger.info(f" Hybrid similarity: {hybrid_similarity.shape}, range [{hybrid_similarity.min():.3f}, {hybrid_similarity.max():.3f}]")
        return hybrid_similarity

    def compute_all_similarities(
        self,
        features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute all similarity matrices from feature dictionary.

        Args:
            features: Dictionary containing all feature arrays

        Returns:
            Dictionary with all similarity matrices
        """
        logger.info("="*60)
        logger.info("COMPUTING ALL SIMILARITIES")
        logger.info("="*60)

        # Compute individual similarities
        genre_similarity = self.compute_genre_similarity(
            features['genre_features']
        )

        text_similarity = self.compute_text_similarity(
            features['text_features']
        )

        metadata_similarity = self.compute_metadata_similarity(
            features['platform_features'],
            features['type_features'],
            features['language_features']
        )

        # Compute hybrid similarity
        hybrid_similarity = self.compute_hybrid_similarity(
            genre_similarity,
            text_similarity,
            metadata_similarity
        )

        logger.info("="*60)
        logger.info("SIMILARITY COMPUTATION COMPLETE")
        logger.info("="*60)

        return {
            'genre_similarity': genre_similarity,
            'text_similarity': text_similarity,
            'metadata_similarity': metadata_similarity,
            'hybrid_similarity': hybrid_similarity
        }

    def get_similarity_statistics(
        self,
        similarity_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistics for a similarity matrix.

        Args:
            similarity_matrix: Similarity matrix

        Returns:
            Dictionary with statistics
        """
        # Get upper triangle (exclude diagonal and duplicates)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]

        return {
            'mean': float(upper_triangle.mean()),
            'std': float(upper_triangle.std()),
            'min': float(upper_triangle.min()),
            'max': float(upper_triangle.max()),
            'median': float(np.median(upper_triangle))
        }
