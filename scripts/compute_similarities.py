"""
Compute similarity matrices from feature matrices.
This script computes content-based similarity scores for recommendations.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging

import numpy as np
from scipy.sparse import load_npz

from tvbingefriend_recommendation_service.ml.similarity_computer import SimilarityComputer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_features(input_dir: Path) -> dict:
    """
    Load feature matrices from disk.

    Args:
        input_dir: Directory containing feature files

    Returns:
        Dictionary with feature arrays
    """
    logger.info("=" * 70)
    logger.info("LOADING FEATURES")
    logger.info("=" * 70)

    required_files = [
        "genre_features.npy",
        "text_features.npz",
        "platform_features.npy",
        "type_features.npy",
        "language_features.npy",
    ]

    # Check all files exist
    for filename in required_files:
        filepath = input_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Feature file not found: {filepath}\n" "Run compute_features.py first."
            )

    # Load features
    genre_features = np.load(input_dir / "genre_features.npy")
    logger.info(f"✓ Loaded genre_features: {genre_features.shape}")

    text_features = load_npz(input_dir / "text_features.npz")
    logger.info(f"✓ Loaded text_features: {text_features.shape}")

    platform_features = np.load(input_dir / "platform_features.npy")
    logger.info(f"✓ Loaded platform_features: {platform_features.shape}")

    type_features = np.load(input_dir / "type_features.npy")
    logger.info(f"✓ Loaded type_features: {type_features.shape}")

    language_features = np.load(input_dir / "language_features.npy")
    logger.info(f"✓ Loaded language_features: {language_features.shape}")

    return {
        "genre_features": genre_features,
        "text_features": text_features,
        "platform_features": platform_features,
        "type_features": type_features,
        "language_features": language_features,
    }


def compute_similarities(
    features: dict,
    genre_weight: float = 0.4,
    text_weight: float = 0.5,
    metadata_weight: float = 0.1,
) -> dict:
    """
    Compute all similarity matrices.

    Args:
        features: Dictionary with feature arrays
        genre_weight: Weight for genre similarity
        text_weight: Weight for text similarity
        metadata_weight: Weight for metadata similarity

    Returns:
        Dictionary with similarity matrices
    """
    logger.info("=" * 70)
    logger.info("COMPUTING SIMILARITIES")
    logger.info("=" * 70)

    # Initialize similarity computer
    computer = SimilarityComputer(
        genre_weight=genre_weight, text_weight=text_weight, metadata_weight=metadata_weight
    )

    # Compute all similarities
    similarities = computer.compute_all_similarities(features)

    # Log statistics for each similarity type
    logger.info("\n" + "=" * 70)
    logger.info("SIMILARITY STATISTICS")
    logger.info("=" * 70)

    for sim_name in [
        "genre_similarity",
        "text_similarity",
        "metadata_similarity",
        "hybrid_similarity",
    ]:
        stats = computer.get_similarity_statistics(similarities[sim_name])
        logger.info(f"\n{sim_name}:")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Std:  {stats['std']:.4f}")
        logger.info(f"  Min:  {stats['min']:.4f}")
        logger.info(f"  Max:  {stats['max']:.4f}")
        logger.info(f"  Median: {stats['median']:.4f}")

    return similarities


def save_similarities(similarities: dict, output_dir: Path, save_to_disk: bool = False):
    """
    Optionally save similarity matrices to disk.

    By default, similarities are NOT saved to reduce storage costs.
    They will be computed in-memory and stored in the database only.

    Args:
        similarities: Dictionary with similarity matrices
        output_dir: Output directory
        save_to_disk: If True, save full matrices (for analysis/debugging only)
    """
    logger.info("\n" + "=" * 70)
    logger.info("SIMILARITY STORAGE")
    logger.info("=" * 70)

    if not save_to_disk:
        logger.info("⊘ Skipping similarity matrix storage (storage optimization)")
        logger.info("  Matrices computed in-memory and will be stored in database only")
        logger.info("  This saves ~205 GB of blob storage for 80k shows")
        logger.info("  Use --save-similarities flag to save for analysis/debugging")
        return

    # Save to disk (only for development/analysis)
    logger.info("⚠️  Saving full similarity matrices to disk")
    logger.info("  Note: This is not recommended for production (large storage cost)")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each similarity matrix
    for name, matrix in similarities.items():
        output_path = output_dir / f"{name}.npy"
        np.save(output_path, matrix)
        file_size = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Saved {name}.npy ({file_size:.2f} MB)")

    # Calculate total size
    total_size = sum((output_dir / f"{name}.npy").stat().st_size for name in similarities.keys())

    logger.info(f"\n✓ All similarities saved to {output_dir}")
    logger.info(f"  Total size: {total_size / 1024 / 1024:.2f} MB")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compute similarity matrices from feature matrices"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Input directory with feature matrices (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for similarities (default: data/processed)",
    )
    parser.add_argument(
        "--genre-weight", type=float, default=0.4, help="Weight for genre similarity (default: 0.4)"
    )
    parser.add_argument(
        "--text-weight", type=float, default=0.5, help="Weight for text similarity (default: 0.5)"
    )
    parser.add_argument(
        "--metadata-weight",
        type=float,
        default=0.1,
        help="Weight for metadata similarity (default: 0.1)",
    )
    parser.add_argument(
        "--save-similarities",
        action="store_true",
        help="Save full similarity matrices to disk (not recommended for production, large storage)",
    )

    args = parser.parse_args()

    # Validate weights
    total_weight = args.genre_weight + args.text_weight + args.metadata_weight
    if total_weight <= 0:
        logger.error("Error: Sum of weights must be greater than 0")
        sys.exit(1)

    # Set up paths
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir

    logger.info("=" * 70)
    logger.info("SIMILARITY COMPUTATION - PRODUCTION")
    logger.info("=" * 70)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Genre weight: {args.genre_weight}")
    logger.info(f"Text weight: {args.text_weight}")
    logger.info(f"Metadata weight: {args.metadata_weight}")
    logger.info(f"Save to disk: {args.save_similarities}")
    logger.info("=" * 70)

    try:
        # Step 1: Load features
        features = load_features(input_dir)

        # Step 2: Compute similarities
        similarities = compute_similarities(
            features=features,
            genre_weight=args.genre_weight,
            text_weight=args.text_weight,
            metadata_weight=args.metadata_weight,
        )

        # Step 3: Save similarities (optional, disabled by default for production)
        save_similarities(similarities, output_dir, save_to_disk=args.save_similarities)

        logger.info("\n" + "=" * 70)
        logger.info("✓ SIMILARITY COMPUTATION COMPLETE")
        logger.info("=" * 70)

        # Return similarities for use by populate_database script
        return similarities

    except Exception as e:
        logger.error(f"Error during similarity computation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
