"""
Fetch all TV show data from the API and prepare it for processing.
This script fetches ALL shows (not limited to 100) for production use.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging
import pandas as pd
import argparse
from tvbingefriend_recommendation_service.services.data_loader_service import ShowDataLoader
from tvbingefriend_recommendation_service.ml.text_processor import clean_html

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def fetch_all_shows(
    show_service_url: str = None,
    batch_size: int = 1000,
    max_shows: int = None
) -> pd.DataFrame:
    """
    Fetch all shows from the API and prepare as DataFrame.

    Args:
        show_service_url: Optional custom show service URL
        batch_size: Number of shows to fetch per batch
        max_shows: Optional limit (for testing, None for production)

    Returns:
        DataFrame with all shows
    """
    logger.info("="*70)
    logger.info("FETCHING TV SHOW DATA")
    logger.info("="*70)

    # Initialize data loader
    loader = ShowDataLoader(show_service_url=show_service_url)

    # Fetch all shows
    shows = loader.get_all_shows(batch_size=batch_size, max_shows=max_shows)

    logger.info(f"✓ Fetched {len(shows)} shows from API")

    # Convert to DataFrame
    df = pd.DataFrame(shows)

    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df


def prepare_show_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare and clean show data for feature extraction.

    Args:
        df: Raw shows DataFrame

    Returns:
        Cleaned DataFrame with essential columns
    """
    logger.info("="*70)
    logger.info("PREPARING SHOW DATA")
    logger.info("="*70)

    # Extract nested fields
    logger.info("Extracting nested fields...")

    # Rating
    df['rating_avg'] = df['rating'].apply(
        lambda x: x.get('average') if isinstance(x, dict) else None
    )

    # Network/Platform
    df['network_name'] = df['network'].apply(
        lambda x: x.get('name') if isinstance(x, dict) else None
    )
    df['webchannel_name'] = df['webchannel'].apply(
        lambda x: x.get('name') if isinstance(x, dict) else None
    )
    df['platform'] = df['network_name'].fillna(df['webchannel_name'])

    # Clean summary text
    logger.info("Cleaning summary text...")
    df['summary_clean'] = df['summary'].apply(clean_html)

    # Select essential columns
    essential_columns = [
        'id', 'name', 'genres', 'summary', 'summary_clean',
        'type', 'language', 'status', 'platform', 'rating_avg'
    ]

    df_clean = df[essential_columns].copy()

    logger.info(f"✓ Prepared {len(df_clean)} shows")
    logger.info(f"Cleaned DataFrame columns: {df_clean.columns.tolist()}")

    # Data quality report
    logger.info("\n" + "="*70)
    logger.info("DATA QUALITY REPORT")
    logger.info("="*70)
    logger.info(f"Total shows: {len(df_clean)}")

    if len(df_clean) > 0:
        logger.info(f"\nFeature availability:")
        logger.info(f"  Genres: {(df_clean['genres'].apply(len) > 0).sum()} shows ({(df_clean['genres'].apply(len) > 0).sum()/len(df_clean)*100:.1f}%)")
        logger.info(f"  Summaries: {(df_clean['summary_clean'].str.len() > 0).sum()} shows ({(df_clean['summary_clean'].str.len() > 0).sum()/len(df_clean)*100:.1f}%)")
        logger.info(f"  Ratings: {df_clean['rating_avg'].notna().sum()} shows ({df_clean['rating_avg'].notna().sum()/len(df_clean)*100:.1f}%)")
        logger.info(f"  Platform: {df_clean['platform'].notna().sum()} shows ({df_clean['platform'].notna().sum()/len(df_clean)*100:.1f}%)")
        logger.info(f"  Type: {df_clean['type'].notna().sum()} shows ({df_clean['type'].notna().sum()/len(df_clean)*100:.1f}%)")
        logger.info(f"  Language: {df_clean['language'].notna().sum()} shows ({df_clean['language'].notna().sum()/len(df_clean)*100:.1f}%)")
    else:
        logger.warning("No shows to report on")

    return df_clean


def save_prepared_data(df: pd.DataFrame, output_dir: Path):
    """
    Save prepared data to CSV.

    Args:
        df: Prepared DataFrame
        output_dir: Output directory
    """
    logger.info("="*70)
    logger.info("SAVING PREPARED DATA")
    logger.info("="*70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_path = output_dir / 'shows_metadata.csv'
    df.to_csv(output_path, index=False)

    logger.info(f"✓ Saved {len(df)} shows to {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Fetch and prepare TV show data from API'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for prepared data (default: data/processed)'
    )
    parser.add_argument(
        '--show-service-url',
        type=str,
        default=None,
        help='Show service URL (default: from config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for fetching shows (default: 1000)'
    )
    parser.add_argument(
        '--max-shows',
        type=int,
        default=None,
        help='Maximum number of shows to fetch (default: None = all shows)'
    )

    args = parser.parse_args()

    # Set up paths
    output_dir = project_root / args.output_dir

    logger.info("="*70)
    logger.info("FETCH AND PREPARE DATA - PRODUCTION")
    logger.info("="*70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Show service URL: {args.show_service_url or 'from config'}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max shows: {args.max_shows or 'ALL'}")
    logger.info("="*70)

    try:
        # Step 1: Fetch all shows
        df_raw = fetch_all_shows(
            show_service_url=args.show_service_url,
            batch_size=args.batch_size,
            max_shows=args.max_shows
        )

        # Step 2: Prepare data
        df_prepared = prepare_show_data(df_raw)

        # Step 3: Save prepared data
        save_prepared_data(df_prepared, output_dir)

        logger.info("\n" + "="*70)
        logger.info("✓ DATA PREPARATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Prepared {len(df_prepared)} shows")
        logger.info(f"Output: {output_dir / 'shows_metadata.csv'}")

    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
