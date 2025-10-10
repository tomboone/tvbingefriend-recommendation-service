"""Service for content-based TV show recommendations."""
import math
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import logging
import tempfile

from tvbingefriend_recommendation_service.repos import SimilarityRepository, MetadataRepository
from tvbingefriend_recommendation_service.models.database import SessionLocal
from tvbingefriend_recommendation_service.storage import BlobStorageClient
from tvbingefriend_recommendation_service.config import (
    use_blob_storage,
    get_azure_storage_connection_string,
    get_storage_container_name
)

logger = logging.getLogger(__name__)


# noinspection PyMethodMayBeStatic
class ContentBasedRecommendationService:
    """
    Service for content-based TV show recommendations.
    Loads pre-computed similarity matrices and serves recommendations.
    """

    def __init__(
            self,
            processed_data_dir: Optional[Path] = None,
            genre_weight: float = 0.4,
            text_weight: float = 0.5,
            metadata_weight: float = 0.1,
            use_blob: Optional[bool] = None,
            blob_prefix: str = "processed"
    ):
        """
        Initialize the recommendation service.

        Args:
            processed_data_dir: Directory containing processed features/similarities (local mode)
            genre_weight: Weight for genre similarity
            text_weight: Weight for text similarity
            metadata_weight: Weight for metadata similarity
            use_blob: Force use of blob storage (None = auto-detect from config)
            blob_prefix: Blob prefix/folder for processed data
        """
        if processed_data_dir is None:
            # Default to data/processed in project root
            project_root = Path(__file__).resolve().parent.parent.parent
            processed_data_dir = project_root / 'data' / 'processed'

        self.processed_data_dir = Path(processed_data_dir)
        self.genre_weight = genre_weight
        self.text_weight = text_weight
        self.metadata_weight = metadata_weight
        self.blob_prefix = blob_prefix

        # Determine if we should use blob storage
        if use_blob is None:
            self.use_blob = use_blob_storage()
        else:
            self.use_blob = use_blob

        # Initialize blob client if needed
        self.blob_client: Optional[BlobStorageClient] = None
        if self.use_blob:
            conn_str = get_azure_storage_connection_string()
            container = get_storage_container_name()
            self.blob_client = BlobStorageClient(
                connection_string=conn_str,
                container_name=container
            )
            logger.info(f"Using blob storage (container: {container}, prefix: {blob_prefix})")
        else:
            logger.info(f"Using local storage: {self.processed_data_dir}")

        # Lazy-load similarity matrices
        self._genre_similarity = None
        self._text_similarity = None
        self._metadata_similarity = None
        self._hybrid_similarity = None
        self._show_id_to_index = None
        self._index_to_show_id = None
        self._temp_dir = None  # For blob downloads

        logger.info("Initialized ContentBasedRecommendationService")
        logger.info(f"Weights - Genre: {genre_weight}, Text: {text_weight}, Metadata: {metadata_weight}")

    def _get_data_dir(self) -> Path:
        """Get the directory containing data files (download from blob if needed)."""
        if not self.use_blob:
            return self.processed_data_dir

        # Using blob storage - download to temp directory if not already done
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="recommendation_data_"))
            logger.info(f"Downloading data from blob storage to {self._temp_dir}...")

            # Download all files from blob
            self.blob_client.download_directory(
                blob_prefix=self.blob_prefix,
                local_dir=self._temp_dir
            )

        return self._temp_dir

    def _load_similarity_matrices(self):
        """Load pre-computed similarity matrices from disk or blob storage."""
        if self._hybrid_similarity is not None:
            return  # Already loaded

        logger.info("Loading similarity matrices...")

        # Get data directory (will download from blob if needed)
        data_dir = self._get_data_dir()

        # Load individual similarity matrices
        self._genre_similarity = np.load(
            data_dir / 'genre_similarity.npy'
        )
        self._text_similarity = np.load(
            data_dir / 'text_similarity.npy'
        )
        self._metadata_similarity = np.load(
            data_dir / 'metadata_similarity.npy'
        )

        # Compute weighted hybrid similarity
        total_weight = self.genre_weight + self.text_weight + self.metadata_weight
        self._hybrid_similarity = (
                (self.genre_weight / total_weight) * self._genre_similarity +
                (self.text_weight / total_weight) * self._text_similarity +
                (self.metadata_weight / total_weight) * self._metadata_similarity
        )

        logger.info(f"✓ Loaded similarity matrices: {self._hybrid_similarity.shape}")

    def _load_show_mappings(self):
        """Load show ID to matrix index mappings."""
        if self._show_id_to_index is not None:
            return  # Already loaded

        import pandas as pd

        # Get data directory (will download from blob if needed)
        data_dir = self._get_data_dir()

        # Load show metadata
        metadata_df = pd.read_csv(data_dir / 'shows_metadata.csv')

        # Create mappings
        self._show_id_to_index = {
            show_id: idx
            for idx, show_id in enumerate(metadata_df['id'])
        }
        self._index_to_show_id = {
            idx: show_id
            for show_id, idx in self._show_id_to_index.items()
        }

        logger.info(f"✓ Loaded mappings for {len(self._show_id_to_index)} shows")

    def get_recommendations_from_matrix(
            self,
            show_id: int,
            n: int = 10,
            min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Get recommendations directly from similarity matrix (no database).

        Args:
            show_id: Source show ID
            n: Number of recommendations
            min_similarity: Minimum similarity threshold

        Returns:
            List of dicts with show_id and similarity_score
        """
        # Lazy load
        self._load_similarity_matrices()
        self._load_show_mappings()

        # Get matrix index for this show
        if show_id not in self._show_id_to_index:
            logger.warning(f"Show ID {show_id} not found in similarity matrix")
            return []

        show_idx = self._show_id_to_index[show_id]

        # Get similarity scores for this show
        similarity_scores = self._hybrid_similarity[show_idx]

        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Exclude the show itself and filter by minimum similarity
        recommendations = []
        for idx in sorted_indices:
            if idx == show_idx:
                continue  # Skip self

            similar_show_id = self._index_to_show_id[idx]
            similarity_score = float(similarity_scores[idx])

            if similarity_score < min_similarity:
                break  # Scores are sorted, so we can break early

            recommendations.append({
                'show_id': similar_show_id,
                'similarity_score': similarity_score,
                'genre_score': float(self._genre_similarity[show_idx, idx]),
                'text_score': float(self._text_similarity[show_idx, idx]),
                'metadata_score': float(self._metadata_similarity[show_idx, idx])
            })

            if len(recommendations) >= n:
                break

        return recommendations

    def get_recommendations_from_db(
            self,
            show_id: int,
            n: int = 10,
            min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Get recommendations from database (pre-computed and stored).

        Args:
            show_id: Source show ID
            n: Number of recommendations
            min_similarity: Minimum similarity threshold

        Returns:
            List of dicts with show metadata and similarity scores
        """
        db = SessionLocal()
        try:
            repo = SimilarityRepository(db)
            recommendations = repo.get_similar_shows_with_metadata(
                show_id=show_id,
                n=n,
                min_similarity=min_similarity
            )
            return recommendations
        finally:
            db.close()

    def compute_and_store_all_similarities(
            self,
            top_n_per_show: int = 20,
            min_similarity: float = 0.1
    ) -> Dict:
        """
        Compute similarities for all shows and store in database.

        Args:
            top_n_per_show: Number of top similar shows to store per show
            min_similarity: Minimum similarity to store

        Returns:
            Dict with statistics
        """
        logger.info("="*60)
        logger.info("COMPUTING AND STORING ALL SIMILARITIES")
        logger.info("="*60)

        # Load matrices
        self._load_similarity_matrices()
        self._load_show_mappings()

        # Compute recommendations for all shows
        all_similarities = {}
        total_shows = len(self._show_id_to_index)

        logger.info(f"Computing similarities for {total_shows} shows...")

        for show_id in self._show_id_to_index.keys():
            recommendations = self.get_recommendations_from_matrix(
                show_id=show_id,
                n=top_n_per_show,
                min_similarity=min_similarity
            )

            if recommendations:
                all_similarities[show_id] = [
                    {
                        'similar_show_id': rec['show_id'],
                        'similarity_score': rec['similarity_score'],
                        'genre_score': rec['genre_score'],
                        'text_score': rec['text_score'],
                        'metadata_score': rec['metadata_score']
                    }
                    for rec in recommendations
                ]

            if (len(all_similarities) % 50) == 0:
                logger.info(f"  Processed {len(all_similarities)}/{total_shows} shows...")

        logger.info(f"✓ Computed similarities for {len(all_similarities)} shows")

        # Store in database
        logger.info("Storing similarities in database...")
        db = SessionLocal()
        try:
            repo = SimilarityRepository(db)
            total_records = repo.bulk_store_all_similarities(all_similarities)

            # Get stats
            stats = repo.get_similarity_stats()
            stats['computed_shows'] = len(all_similarities)
            stats['top_n_per_show'] = top_n_per_show
            stats['min_similarity'] = min_similarity

            logger.info("="*60)
            logger.info("SIMILARITY COMPUTATION COMPLETE")
            logger.info("="*60)
            logger.info(f"Total records stored: {total_records}")
            logger.info(f"Unique shows: {stats['unique_shows']}")
            logger.info(f"Avg per show: {stats['avg_similarities_per_show']:.1f}")

            return stats

        finally:
            db.close()

    def sync_metadata_to_db(self, shows_data: List[Dict]) -> int:
        """
        Sync show metadata to database for quick lookups.

        Args:
            shows_data: List of show data dicts

        Returns:
            Number of shows stored
        """
        logger.info(f"Syncing {len(shows_data)} shows to database...")

        db = SessionLocal()
        try:
            repo = MetadataRepository(db)

            # Format data for repository
            formatted_data = []
            for show in shows_data:
                # Handle NaN values - convert to None for database
                rating = show.get("rating_avg")
                if rating is not None and (
                    isinstance(rating, float) and math.isnan(rating)
                ):
                    rating = None
                formatted_data.append({
                    'show_id': show['id'],
                    'name': show['name'],
                    'genres': show.get('genres'),
                    'summary': show.get('summary_clean') or show.get('summary'),
                    'rating': rating,
                    'type': show.get('type'),
                    'language': show.get('language'),
                    'network': show.get('platform')
                })

            count = repo.bulk_store_shows(formatted_data)
            logger.info(f"✓ Synced {count} shows to database")
            return count

        finally:
            db.close()

    def get_stats(self) -> Dict:
        """Get statistics about the recommendation system."""
        db = SessionLocal()
        try:
            similarity_repo = SimilarityRepository(db)
            metadata_repo = MetadataRepository(db)

            similarity_stats = similarity_repo.get_similarity_stats()
            metadata_count = metadata_repo.count_shows()

            return {
                'similarity_stats': similarity_stats,
                'cached_shows': metadata_count,
                'weights': {
                    'genre': self.genre_weight,
                    'text': self.text_weight,
                    'metadata': self.metadata_weight
                }
            }
        finally:
            db.close()
