"""Repository for managing show similarity data in the database."""

from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from datetime import datetime, UTC
import logging

from tvbingefriend_recommendation_service.models import ShowSimilarity
from tvbingefriend_recommendation_service.models import ShowMetadata

logger = logging.getLogger(__name__)


class SimilarityRepository:
    """
    Repository for managing show similarity data in the database.
    """

    def __init__(self, db: Session):
        self.db = db

    def store_similarities(
            self,
            show_id: int,
            similar_shows: List[Dict],
            batch_size: int = 100
    ) -> int:
        """
        Store similarity scores for a show.

        Args:
            show_id: Source show ID
            similar_shows: List of dicts with keys:
                - similar_show_id: int
                - similarity_score: float
                - genre_score: float (optional)
                - text_score: float (optional)
                - metadata_score: float (optional)
            batch_size: Number of records to insert per batch

        Returns:
            Number of similarities stored
        """
        # Delete existing similarities for this show
        self.db.query(ShowSimilarity).filter(
            ShowSimilarity.show_id == show_id
        ).delete()

        # Prepare records
        records = []
        for item in similar_shows:
            record = ShowSimilarity(
                show_id=show_id,
                similar_show_id=item['similar_show_id'],
                similarity_score=item['similarity_score'],
                genre_score=item.get('genre_score'),
                text_score=item.get('text_score'),
                metadata_score=item.get('metadata_score'),
                computed_at=datetime.now(UTC)
            )
            records.append(record)

        # Batch insert
        count = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            self.db.bulk_save_objects(batch)
            self.db.commit()
            count += len(batch)

            if (i + batch_size) % 1000 == 0:
                logger.info(f"Stored {count} similarities for show {show_id}...")

        logger.info(f"✓ Stored {count} similarities for show {show_id}")
        return count

    def bulk_store_all_similarities(
            self,
            all_similarities: Dict[int, List[Dict]],
            batch_size: int = 1000,
            clear_existing: bool = True
    ) -> int:
        """
        Store similarities for multiple shows in bulk.

        Args:
            all_similarities: Dict mapping show_id to list of similar shows
            batch_size: Number of records to insert per batch
            clear_existing: Whether to clear existing similarities before inserting

        Returns:
            Total number of similarities stored
        """
        # Clear all existing similarities if requested
        if clear_existing:
            logger.info("Clearing existing similarities...")
            self.db.query(ShowSimilarity).delete()
            self.db.commit()

        # Prepare all records
        all_records = []
        for show_id, similar_shows in all_similarities.items():
            for item in similar_shows:
                record = ShowSimilarity(
                    show_id=show_id,
                    similar_show_id=item['similar_show_id'],
                    similarity_score=item['similarity_score'],
                    genre_score=item.get('genre_score'),
                    text_score=item.get('text_score'),
                    metadata_score=item.get('metadata_score'),
                    computed_at=datetime.now(UTC)
                )
                all_records.append(record)

        # Batch insert
        total_count = 0
        logger.info(f"Inserting {len(all_records)} similarity records...")

        for i in range(0, len(all_records), batch_size):
            batch = all_records[i:i + batch_size]
            self.db.bulk_save_objects(batch)
            self.db.commit()
            total_count += len(batch)

            if (i + batch_size) % 5000 == 0:
                logger.info(f"  Inserted {total_count}/{len(all_records)} records...")

        logger.info(f"✓ Stored {total_count} total similarities for {len(all_similarities)} shows")
        return total_count

    # noinspection PyTypeChecker
    def get_similar_shows(
            self,
            show_id: int,
            n: int = 10,
            min_similarity: float = 0.0
    ) -> List[ShowSimilarity]:
        """
        Get most similar shows for a given show.

        Args:
            show_id: Source show ID
            n: Number of recommendations to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of ShowSimilarity objects
        """
        return (
            self.db.query(ShowSimilarity)
            .filter(
                and_(
                    ShowSimilarity.show_id == show_id,
                    ShowSimilarity.similarity_score >= min_similarity
                )
            )
            .order_by(desc(ShowSimilarity.similarity_score))
            .limit(n)
            .all()
        )

    def get_similar_shows_with_metadata(
            self,
            show_id: int,
            n: int = 10,
            min_similarity: float = 0.0
    ) -> List[Dict]:
        """
        Get similar shows with their metadata.

        Args:
            show_id: Source show ID
            n: Number of recommendations
            min_similarity: Minimum similarity threshold

        Returns:
            List of dicts with similarity and show metadata
        """
        results = (
            self.db.query(ShowSimilarity, ShowMetadata)
            .join(
                ShowMetadata,
                ShowSimilarity.similar_show_id == ShowMetadata.show_id
            )
            .filter(
                and_(
                    ShowSimilarity.show_id == show_id,
                    ShowSimilarity.similarity_score >= min_similarity
                )
            )
            .order_by(desc(ShowSimilarity.similarity_score))
            .limit(n)
            .all()
        )

        # Format results
        recommendations = []
        for similarity, metadata in results:
            recommendations.append({
                'show_id': metadata.show_id,
                'name': metadata.name,
                'genres': metadata.genres,
                'summary': metadata.summary,
                'rating': metadata.rating,
                'type': metadata.type,
                'language': metadata.language,
                'network': metadata.network,
                'similarity_score': similarity.similarity_score,
                'genre_score': similarity.genre_score,
                'text_score': similarity.text_score,
                'metadata_score': similarity.metadata_score,
            })

        return recommendations

    # noinspection PyTypeChecker
    def get_all_show_ids_with_similarities(self) -> List[int]:
        """Get list of all show IDs that have computed similarities."""
        result = (
            self.db.query(ShowSimilarity.show_id)
            .distinct()
            .all()
        )
        return [row[0] for row in result]

    def count_similarities(self, show_id: Optional[int] = None) -> int:
        """
        Count similarity records.

        Args:
            show_id: If provided, count for specific show. Otherwise count all.

        Returns:
            Number of similarity records
        """
        query = self.db.query(ShowSimilarity)

        if show_id is not None:
            query = query.filter(ShowSimilarity.show_id == show_id)

        return query.count()

    def delete_similarities(self, show_id: int) -> int:
        """
        Delete all similarities for a show.

        Args:
            show_id: Show ID to delete similarities for

        Returns:
            Number of deleted records
        """
        count = (
            self.db.query(ShowSimilarity)
            .filter(ShowSimilarity.show_id == show_id)
            .delete()
        )
        self.db.commit()

        logger.info(f"Deleted {count} similarities for show {show_id}")
        return count

    def get_similarity_stats(self) -> Dict:
        """Get statistics about stored similarities."""
        total_records = self.db.query(ShowSimilarity).count()
        unique_shows = (
            self.db.query(ShowSimilarity.show_id)
            .distinct()
            .count()
        )

        # Average similarities per show
        avg_per_show = total_records / unique_shows if unique_shows > 0 else 0

        # Get latest computation time
        latest = (
            self.db.query(ShowSimilarity.computed_at)
            .order_by(desc(ShowSimilarity.computed_at))
            .first()
        )

        return {
            'total_records': total_records,
            'unique_shows': unique_shows,
            'avg_similarities_per_show': avg_per_show,
            'last_computed': latest[0] if latest else None
        }
