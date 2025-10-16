"""
Script to compute similarities and populate the database.
Run this after training your model or when you want to refresh recommendations.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging

import numpy as np
import pandas as pd

from tvbingefriend_recommendation_service.services import ContentBasedRecommendationService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("POPULATING RECOMMENDATION DATABASE")
    logger.info("=" * 70)

    # Initialize service
    service = ContentBasedRecommendationService()

    # Step 1: Sync show metadata to database
    logger.info("\nStep 1: Syncing show metadata to database...")
    processed_dir = project_root / "data" / "processed"
    shows_df = pd.read_csv(processed_dir / "shows_metadata.csv")

    # Clean DataFrame - replace NaN with None
    logger.info(f"Loaded {len(shows_df)} shows from CSV")
    shows_df = clean_dataframe_for_db(shows_df)

    shows_data = shows_df.to_dict("records")

    metadata_count = service.sync_metadata_to_db(shows_data)
    logger.info(f"✓ Synced {metadata_count} shows")

    # Step 2: Compute and store all similarities
    logger.info("\nStep 2: Computing and storing similarities...")
    stats = service.compute_and_store_all_similarities()

    # Step 3: Display results
    logger.info("\n" + "=" * 70)
    logger.info("POPULATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total similarity records: {stats['total_records']}")
    logger.info(f"Unique shows with recommendations: {stats['unique_shows']}")
    logger.info(f"Average similarities per show: {stats['avg_similarities_per_show']:.1f}")
    logger.info(f"Last computed: {stats['last_computed']}")

    # Step 4: Test a few recommendations
    logger.info("\n" + "=" * 70)
    logger.info("TESTING RECOMMENDATIONS")
    logger.info("=" * 70)

    test_show_ids = shows_df["id"].head(3).tolist()

    for show_id in test_show_ids:
        show_name = shows_df[shows_df["id"] == show_id].iloc[0]["name"]
        logger.info(f"\nRecommendations for '{show_name}' (ID: {show_id}):")

        recommendations = service.get_recommendations_from_db(show_id=show_id, n=5)

        for i, rec in enumerate(recommendations, 1):
            logger.info(
                f"  {i}. {rec['name']} "
                f"(score: {rec['similarity_score']:.3f}, "
                f"genres: {rec['genres']})"
            )

    logger.info("\n✓ Database population complete!")


if __name__ == "__main__":
    main()
