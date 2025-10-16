"""
Compute feature matrices from prepared show data.
This script performs feature engineering for content-based recommendations.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from tvbingefriend_recommendation_service.ml.feature_extractor import FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_prepared_data(input_dir: Path) -> pd.DataFrame:
    """
    Load prepared show data.

    Args:
        input_dir: Directory containing shows_metadata.csv

    Returns:
        DataFrame with show data
    """
    logger.info("=" * 70)
    logger.info("LOADING PREPARED DATA")
    logger.info("=" * 70)

    metadata_path = input_dir / "shows_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {metadata_path}\n" "Run fetch_and_prepare_data.py first."
        )

    df = pd.read_csv(metadata_path)
    logger.info(f"✓ Loaded {len(df)} shows from {metadata_path}")

    return df


def extract_features(
    df: pd.DataFrame,
    max_text_features: int = 500,
    text_min_df: int = 2,
    text_max_df: float = 0.8,
    top_n_platforms: int = 20,
    top_n_languages: int = 5,
) -> dict:
    """
    Extract all features from show data.

    Args:
        df: DataFrame with show data
        max_text_features: Max TF-IDF features
        text_min_df: Min document frequency for TF-IDF
        text_max_df: Max document frequency for TF-IDF
        top_n_platforms: Number of top platforms to encode
        top_n_languages: Number of top languages to encode

    Returns:
        Dictionary with features and encoders
    """
    logger.info("=" * 70)
    logger.info("EXTRACTING FEATURES")
    logger.info("=" * 70)

    # Initialize feature extractor
    extractor = FeatureExtractor(
        max_text_features=max_text_features,
        text_min_df=text_min_df,
        text_max_df=text_max_df,
        top_n_platforms=top_n_platforms,
        top_n_languages=top_n_languages,
    )

    # Extract all features
    features = extractor.extract_all_features(df)

    return features


def save_features(features: dict, output_dir: Path):
    """
    Save feature matrices and encoders to disk.

    Args:
        features: Dictionary with features and encoders
        output_dir: Output directory
    """
    logger.info("=" * 70)
    logger.info("SAVING FEATURES")
    logger.info("=" * 70)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save feature matrices
    np.save(output_dir / "genre_features.npy", features["genre_features"])
    logger.info("✓ Saved genre_features.npy")

    # Save text features as sparse matrix (more efficient)
    save_npz(output_dir / "text_features.npz", features["text_features"])
    logger.info("✓ Saved text_features.npz")

    np.save(output_dir / "platform_features.npy", features["platform_features"])
    logger.info("✓ Saved platform_features.npy")

    np.save(output_dir / "type_features.npy", features["type_features"])
    logger.info("✓ Saved type_features.npy")

    np.save(output_dir / "language_features.npy", features["language_features"])
    logger.info("✓ Saved language_features.npy")

    # Save encoders/vectorizers for later use
    with open(output_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(features["tfidf_vectorizer"], f)
    logger.info("✓ Saved tfidf_vectorizer.pkl")

    with open(output_dir / "genre_encoder.pkl", "wb") as f:
        pickle.dump(features["genre_encoder"], f)
    logger.info("✓ Saved genre_encoder.pkl")

    # Calculate total size
    total_size = sum(
        (output_dir / f).stat().st_size
        for f in [
            "genre_features.npy",
            "text_features.npz",
            "platform_features.npy",
            "type_features.npy",
            "language_features.npy",
            "tfidf_vectorizer.pkl",
            "genre_encoder.pkl",
        ]
    )

    logger.info(f"\n✓ All features saved to {output_dir}")
    logger.info(f"  Total size: {total_size / 1024 / 1024:.2f} MB")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Compute feature matrices from prepared show data")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Input directory with prepared data (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for features (default: data/processed)",
    )
    parser.add_argument(
        "--max-text-features", type=int, default=500, help="Maximum TF-IDF features (default: 500)"
    )
    parser.add_argument(
        "--text-min-df", type=int, default=2, help="Min document frequency for TF-IDF (default: 2)"
    )
    parser.add_argument(
        "--text-max-df",
        type=float,
        default=0.8,
        help="Max document frequency for TF-IDF (default: 0.8)",
    )
    parser.add_argument(
        "--top-n-platforms",
        type=int,
        default=20,
        help="Number of top platforms to encode (default: 20)",
    )
    parser.add_argument(
        "--top-n-languages",
        type=int,
        default=5,
        help="Number of top languages to encode (default: 5)",
    )

    args = parser.parse_args()

    # Set up paths
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir

    logger.info("=" * 70)
    logger.info("FEATURE COMPUTATION - PRODUCTION")
    logger.info("=" * 70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Max text features: {args.max_text_features}")
    logger.info(f"Text min_df: {args.text_min_df}")
    logger.info(f"Text max_df: {args.text_max_df}")
    logger.info(f"Top N platforms: {args.top_n_platforms}")
    logger.info(f"Top N languages: {args.top_n_languages}")
    logger.info("=" * 70)

    try:
        # Step 1: Load prepared data
        df = load_prepared_data(input_dir)

        # Step 2: Extract features
        features = extract_features(
            df=df,
            max_text_features=args.max_text_features,
            text_min_df=args.text_min_df,
            text_max_df=args.text_max_df,
            top_n_platforms=args.top_n_platforms,
            top_n_languages=args.top_n_languages,
        )

        # Step 3: Save features
        save_features(features, output_dir)

        logger.info("\n" + "=" * 70)
        logger.info("✓ FEATURE COMPUTATION COMPLETE")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error during feature computation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
