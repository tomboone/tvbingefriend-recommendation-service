"""
Populate the database with show metadata and pre-computed similarities.
This script loads processed data and stores it in the MySQL database for fast API retrieval.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import numpy as np
import argparse

from tvbingefriend_recommendation_service.services import ContentBasedRecommendationService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def clean_dataframe_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame by replacing NaN/NA values with None for database compatibility.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Replace various types of missing values with None
    df = df.replace({np.nan: None, pd.NA: None})
    df = df.where(pd.notnull(df), None)

    return df


def load_and_sync_metadata(
    service: ContentBasedRecommendationService,
    input_dir: Path
) -> int:
    """
    Load show metadata and sync to database.

    Args:
        service: Recommendation service instance
        input_dir: Directory containing processed data

    Returns:
        Number of shows synced
    """
    logger.info("="*70)
    logger.info("SYNCING SHOW METADATA TO DATABASE")
    logger.info("="*70)

    metadata_path = input_dir / 'shows_metadata.csv'

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Run fetch_and_prepare_data.py first."
        )

    # Load metadata
    shows_df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(shows_df)} shows from {metadata_path}")

    # Clean DataFrame - replace NaN with None
    shows_df = clean_dataframe_for_db(shows_df)

    # Convert to dict records
    shows_data = shows_df.to_dict('records')

    # Sync to database
    count = service.sync_metadata_to_db(shows_data)
    logger.info(f"✓ Synced {count} shows to database")

    return count


def compute_and_store_similarities(
    input_dir: Path,
    genre_weight: float = 0.4,
    text_weight: float = 0.5,
    metadata_weight: float = 0.1,
    top_n_per_show: int = 20,
    min_similarity: float = 0.1
) -> dict:
    """
    Compute similarities from features (in-memory) and store in database.
    This avoids loading/saving large similarity matrices.

    Args:
        input_dir: Directory with feature files
        genre_weight: Weight for genre similarity
        text_weight: Weight for text similarity
        metadata_weight: Weight for metadata similarity
        top_n_per_show: Number of top similar shows to store per show
        min_similarity: Minimum similarity threshold

    Returns:
        Statistics dictionary
    """
    logger.info("\n" + "="*70)
    logger.info("COMPUTING AND STORING SIMILARITIES")
    logger.info("="*70)
    logger.info(f"Genre weight: {genre_weight}")
    logger.info(f"Text weight: {text_weight}")
    logger.info(f"Metadata weight: {metadata_weight}")
    logger.info(f"Top N per show: {top_n_per_show}")
    logger.info(f"Min similarity: {min_similarity}")

    # Import similarity computation modules
    import numpy as np
    from scipy.sparse import load_npz
    from tvbingefriend_recommendation_service.ml.similarity_computer import SimilarityComputer

    # Load features
    logger.info("\nLoading features from disk...")
    genre_features = np.load(input_dir / 'genre_features.npy')
    text_features = load_npz(input_dir / 'text_features.npz')
    platform_features = np.load(input_dir / 'platform_features.npy')
    type_features = np.load(input_dir / 'type_features.npy')
    language_features = np.load(input_dir / 'language_features.npy')

    logger.info(f"✓ Loaded features for {genre_features.shape[0]} shows")

    # Load show IDs from metadata
    import pandas as pd
    metadata_df = pd.read_csv(input_dir / 'shows_metadata.csv')
    show_ids = metadata_df['id'].tolist()

    # Compute similarities one show at a time (memory-efficient for large datasets)
    logger.info("\nComputing similarities (batch processing for memory efficiency)...")
    logger.info(f"Processing {len(show_ids)} shows...")

    computer = SimilarityComputer(
        genre_weight=genre_weight,
        text_weight=text_weight,
        metadata_weight=metadata_weight
    )

    # Initialize database connection for incremental writes
    from tvbingefriend_recommendation_service.repos import SimilarityRepository
    from tvbingefriend_recommendation_service.models.database import SessionLocal

    db = SessionLocal()
    repo = SimilarityRepository(db)

    # Clear existing similarities once at the start
    logger.info("Clearing existing similarities from database...")
    from tvbingefriend_recommendation_service.models import ShowSimilarity
    db.query(ShowSimilarity).delete()
    db.commit()
    logger.info("✓ Database cleared, starting incremental similarity computation...")

    all_similarities = {}
    total_records_stored = 0
    batch_size = 5000  # Write to DB every 5000 shows
    first_batch = True

    from sklearn.metrics.pairwise import cosine_similarity

    for idx, show_id in enumerate(show_ids):
        # Compute similarity for this single show against all others
        # This computes only one row of the similarity matrix at a time
        show_genre = genre_features[idx:idx+1]  # Shape: (1, n_genre_features)
        show_text = text_features[idx:idx+1]    # Shape: (1, n_text_features)
        show_platform = platform_features[idx:idx+1]
        show_type = type_features[idx:idx+1]
        show_language = language_features[idx:idx+1]

        # Compute similarities against all shows
        genre_sim = cosine_similarity(show_genre, genre_features)[0]
        text_sim = cosine_similarity(show_text, text_features)[0]

        # Metadata similarity (platform, type, language combined)
        platform_sim = cosine_similarity(show_platform, platform_features)[0]
        type_sim = cosine_similarity(show_type, type_features)[0]
        language_sim = cosine_similarity(show_language, language_features)[0]
        metadata_sim = (platform_sim + type_sim + language_sim) / 3

        # Hybrid similarity
        hybrid_sim = (genre_weight * genre_sim +
                     text_weight * text_sim +
                     metadata_weight * metadata_sim)

        # Get top N (excluding self)
        top_indices = np.argsort(hybrid_sim)[::-1]

        recommendations = []
        for similar_idx in top_indices:
            if similar_idx == idx:
                continue  # Skip self

            similarity_score = float(hybrid_sim[similar_idx])

            if similarity_score < min_similarity:
                break

            if len(recommendations) >= top_n_per_show:
                break

            recommendations.append({
                'similar_show_id': show_ids[similar_idx],
                'similarity_score': similarity_score,
                'genre_score': float(genre_sim[similar_idx]),
                'text_score': float(text_sim[similar_idx]),
                'metadata_score': float(metadata_sim[similar_idx])
            })

        if recommendations:
            all_similarities[show_id] = recommendations

        # Write to database every batch_size shows
        if ((idx + 1) % batch_size) == 0 or (idx + 1) == len(show_ids):
            if all_similarities:
                # Don't clear existing records after first batch (already cleared at start)
                batch_records = repo.bulk_store_all_similarities(
                    all_similarities,
                    clear_existing=False
                )
                total_records_stored += batch_records
                logger.info(f"  Processed {idx + 1}/{len(show_ids)} shows... (stored {total_records_stored} similarity records)")
                all_similarities = {}  # Clear memory
            else:
                logger.info(f"  Processed {idx + 1}/{len(show_ids)} shows...")
        elif ((idx + 1) % 1000) == 0:
            logger.info(f"  Processed {idx + 1}/{len(show_ids)} shows...")

    logger.info(f"✓ Stored {total_records_stored} total similarity records")

    # Get final stats
    try:
        stats = repo.get_similarity_stats()
        stats['total_records'] = total_records_stored
        stats['top_n_per_show'] = top_n_per_show
        stats['min_similarity'] = min_similarity

        logger.info("="*70)
        logger.info("SIMILARITY STORAGE COMPLETE")
        logger.info("="*70)
        logger.info(f"Total records stored: {total_records_stored}")
        logger.info(f"Unique shows: {stats['unique_shows']}")
        logger.info(f"Avg per show: {stats['avg_similarities_per_show']:.1f}")

        return stats

    finally:
        db.close()


def verify_recommendations(
    service: ContentBasedRecommendationService,
    metadata_path: Path,
    num_tests: int = 3
):
    """
    Verify recommendations for a few shows.

    Args:
        service: Recommendation service instance
        metadata_path: Path to metadata CSV
        num_tests: Number of shows to test
    """
    logger.info("\n" + "="*70)
    logger.info("TESTING RECOMMENDATIONS")
    logger.info("="*70)

    # Load metadata for test
    shows_df = pd.read_csv(metadata_path)
    test_show_ids = shows_df['id'].head(num_tests).tolist()

    for show_id in test_show_ids:
        show_name = shows_df[shows_df['id'] == show_id].iloc[0]['name']
        logger.info(f"\nRecommendations for '{show_name}' (ID: {show_id}):")

        recommendations = service.get_recommendations_from_db(
            show_id=show_id,
            n=5
        )

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                logger.info(
                    f"  {i}. {rec['name']} "
                    f"(score: {rec['similarity_score']:.3f}, "
                    f"genres: {rec['genres']})"
                )
        else:
            logger.warning(f"  No recommendations found")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Populate database with show metadata and similarities'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processed',
        help='Input directory with processed data (default: data/processed)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Top N similar shows to store per show (default: 20)'
    )
    parser.add_argument(
        '--min-similarity',
        type=float,
        default=0.1,
        help='Minimum similarity threshold (default: 0.1)'
    )
    parser.add_argument(
        '--skip-metadata',
        action='store_true',
        help='Skip metadata sync (only compute similarities)'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip recommendation testing'
    )
    parser.add_argument(
        '--genre-weight',
        type=float,
        default=0.4,
        help='Weight for genre similarity (default: 0.4)'
    )
    parser.add_argument(
        '--text-weight',
        type=float,
        default=0.5,
        help='Weight for text similarity (default: 0.5)'
    )
    parser.add_argument(
        '--metadata-weight',
        type=float,
        default=0.1,
        help='Weight for metadata similarity (default: 0.1)'
    )

    args = parser.parse_args()

    # Set up paths
    input_dir = project_root / args.input_dir

    logger.info("="*70)
    logger.info("POPULATE DATABASE - PRODUCTION")
    logger.info("="*70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Genre weight: {args.genre_weight}")
    logger.info(f"Text weight: {args.text_weight}")
    logger.info(f"Metadata weight: {args.metadata_weight}")
    logger.info(f"Top N per show: {args.top_n}")
    logger.info(f"Min similarity: {args.min_similarity}")
    logger.info(f"Skip metadata sync: {args.skip_metadata}")
    logger.info(f"Skip testing: {args.skip_test}")
    logger.info("="*70)

    try:
        # Initialize service (for testing only)
        service = ContentBasedRecommendationService(
            processed_data_dir=input_dir,
            use_blob=False  # Testing uses database
        )

        metadata_count = 0

        # Step 1: Sync metadata (optional)
        if not args.skip_metadata:
            metadata_count = load_and_sync_metadata(service, input_dir)
        else:
            logger.info("\n⊘ Skipping metadata sync")

        # Step 2: Compute and store similarities (in-memory, no matrix files)
        stats = compute_and_store_similarities(
            input_dir=input_dir,
            genre_weight=args.genre_weight,
            text_weight=args.text_weight,
            metadata_weight=args.metadata_weight,
            top_n_per_show=args.top_n,
            min_similarity=args.min_similarity
        )

        # Step 3: Verify recommendations (optional)
        if not args.skip_test:
            verify_recommendations(
                service=service,
                metadata_path=input_dir / 'shows_metadata.csv',
                num_tests=3
            )
        else:
            logger.info("\n⊘ Skipping recommendation testing")

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("✓ DATABASE POPULATION COMPLETE")
        logger.info("="*70)
        if not args.skip_metadata:
            logger.info(f"Shows synced: {metadata_count}")
        logger.info(f"Similarity records: {stats['total_records']}")
        logger.info(f"Unique shows with recommendations: {stats['unique_shows']}")
        logger.info(f"Average similarities per show: {stats['avg_similarities_per_show']:.1f}")
        logger.info(f"Last computed: {stats['last_computed']}")

    except Exception as e:
        logger.error(f"Error during database population: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
