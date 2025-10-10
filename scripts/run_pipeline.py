"""
Master pipeline script that runs all data processing steps in sequence.
This is the main entry point for the production data pipeline.

Usage:
    # Run full pipeline (fetch, compute, populate)
    python scripts/run_pipeline.py

    # Run specific steps only
    python scripts/run_pipeline.py --steps fetch,features

    # Run with custom parameters
    python scripts/run_pipeline.py --max-shows 1000 --top-n 30
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import logging
import argparse
import subprocess
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def run_step(script_name: str, args: list = None) -> bool:
    """
    Run a pipeline step script.

    Args:
        script_name: Name of the script to run
        args: Optional list of arguments

    Returns:
        True if successful, False otherwise
    """
    script_path = project_root / 'scripts' / script_name
    cmd = [sys.executable, str(script_path)]

    if args:
        cmd.extend(args)

    logger.info(f"Running: {' '.join(cmd)}")
    logger.info("="*70)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info("="*70)
        logger.info(f"✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {script_name} failed with exit code {e.returncode}")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Run the complete recommendation data pipeline'
    )

    # Pipeline control
    parser.add_argument(
        '--steps',
        type=str,
        default='all',
        help='Comma-separated list of steps to run: fetch,features,similarities,populate,all (default: all)'
    )

    # Common parameters
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Data directory for processed files (default: data/processed)'
    )

    # Fetch step parameters
    parser.add_argument(
        '--show-service-url',
        type=str,
        default=None,
        help='Show service URL (default: from config)'
    )
    parser.add_argument(
        '--max-shows',
        type=int,
        default=None,
        help='Maximum number of shows to fetch (default: None = all shows)'
    )

    # Feature extraction parameters
    parser.add_argument(
        '--max-text-features',
        type=int,
        default=500,
        help='Maximum TF-IDF features (default: 500)'
    )

    # Similarity computation parameters
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

    # Database population parameters
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

    args = parser.parse_args()

    # Parse steps
    if args.steps.lower() == 'all':
        # Note: 'populate' now computes similarities in-memory (no separate step needed)
        steps = ['fetch', 'features', 'populate']
    else:
        steps = [s.strip().lower() for s in args.steps.split(',')]

    # Validate steps
    valid_steps = ['fetch', 'features', 'populate']
    invalid_steps = [s for s in steps if s not in valid_steps]
    if invalid_steps:
        logger.error(f"Invalid steps: {invalid_steps}")
        logger.error(f"Valid steps: {valid_steps} (Note: 'similarities' step removed - now computed in 'populate' step)")
        sys.exit(1)

    # Start pipeline
    start_time = time.time()
    logger.info("="*70)
    logger.info("RECOMMENDATION DATA PIPELINE - PRODUCTION")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Steps to run: {', '.join(steps)}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info("="*70)
    logger.info("")

    results = {}

    # Step 1: Fetch and prepare data
    if 'fetch' in steps:
        logger.info("\n" + "▶"*35)
        logger.info("STEP 1: FETCH AND PREPARE DATA")
        logger.info("▶"*35 + "\n")

        fetch_args = [
            '--output-dir', args.data_dir,
        ]
        if args.show_service_url:
            fetch_args.extend(['--show-service-url', args.show_service_url])
        if args.max_shows:
            fetch_args.extend(['--max-shows', str(args.max_shows)])

        results['fetch'] = run_step('fetch_and_prepare_data.py', fetch_args)

        if not results['fetch']:
            logger.error("Pipeline failed at fetch step")
            sys.exit(1)

    # Step 2: Compute features
    if 'features' in steps:
        logger.info("\n" + "▶"*35)
        logger.info("STEP 2: COMPUTE FEATURES")
        logger.info("▶"*35 + "\n")

        features_args = [
            '--input-dir', args.data_dir,
            '--output-dir', args.data_dir,
            '--max-text-features', str(args.max_text_features),
        ]

        results['features'] = run_step('compute_features.py', features_args)

        if not results['features']:
            logger.error("Pipeline failed at features step")
            sys.exit(1)

    # Step 3: Populate database (computes similarities in-memory)
    if 'populate' in steps:
        logger.info("\n" + "▶"*35)
        logger.info("STEP 3: POPULATE DATABASE")
        logger.info("▶"*35 + "\n")

        populate_args = [
            '--input-dir', args.data_dir,
            '--genre-weight', str(args.genre_weight),
            '--text-weight', str(args.text_weight),
            '--metadata-weight', str(args.metadata_weight),
            '--top-n', str(args.top_n),
            '--min-similarity', str(args.min_similarity),
        ]

        results['populate'] = run_step('populate_database.py', populate_args)

        if not results['populate']:
            logger.error("Pipeline failed at populate step")
            sys.exit(1)

    # Pipeline complete
    elapsed_time = time.time() - start_time
    logger.info("\n" + "="*70)
    logger.info("✓ PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time: {elapsed_time / 60:.2f} minutes")
    logger.info("\nStep results:")
    for step, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {step}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
