"""
Tests for scripts/compute_similarities.py
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix, save_npz

# Import functions from the script
from scripts.compute_similarities import (
    compute_similarities,
    load_features,
    main,
    save_similarities,
)


class TestLoadFeatures:
    """Tests for load_features function."""

    def test_load_features_success(self, temp_data_dir):
        """Test successful loading of features."""
        # Create all required feature files
        np.save(temp_data_dir / "genre_features.npy", np.random.rand(5, 10))
        np.save(temp_data_dir / "platform_features.npy", np.random.rand(5, 5))
        np.save(temp_data_dir / "type_features.npy", np.random.rand(5, 3))
        np.save(temp_data_dir / "language_features.npy", np.random.rand(5, 4))
        save_npz(temp_data_dir / "text_features.npz", csr_matrix(np.random.rand(5, 100)))

        result = load_features(temp_data_dir)

        assert "genre_features" in result
        assert "text_features" in result
        assert "platform_features" in result
        assert "type_features" in result
        assert "language_features" in result

        assert result["genre_features"].shape == (5, 10)
        assert result["text_features"].shape == (5, 100)
        assert result["platform_features"].shape == (5, 5)
        assert result["type_features"].shape == (5, 3)
        assert result["language_features"].shape == (5, 4)

    def test_load_features_missing_genre_file(self, temp_data_dir):
        """Test error when genre features file is missing."""
        # Create all files except genre_features
        np.save(temp_data_dir / "platform_features.npy", np.random.rand(5, 5))
        np.save(temp_data_dir / "type_features.npy", np.random.rand(5, 3))
        np.save(temp_data_dir / "language_features.npy", np.random.rand(5, 4))
        save_npz(temp_data_dir / "text_features.npz", csr_matrix(np.random.rand(5, 100)))

        with pytest.raises(FileNotFoundError) as exc_info:
            load_features(temp_data_dir)

        assert "genre_features.npy" in str(exc_info.value)
        assert "Run compute_features.py first" in str(exc_info.value)

    def test_load_features_missing_text_file(self, temp_data_dir):
        """Test error when text features file is missing."""
        np.save(temp_data_dir / "genre_features.npy", np.random.rand(5, 10))
        np.save(temp_data_dir / "platform_features.npy", np.random.rand(5, 5))
        np.save(temp_data_dir / "type_features.npy", np.random.rand(5, 3))
        np.save(temp_data_dir / "language_features.npy", np.random.rand(5, 4))

        with pytest.raises(FileNotFoundError) as exc_info:
            load_features(temp_data_dir)

        assert "text_features.npz" in str(exc_info.value)

    def test_load_features_missing_platform_file(self, temp_data_dir):
        """Test error when platform features file is missing."""
        np.save(temp_data_dir / "genre_features.npy", np.random.rand(5, 10))
        save_npz(temp_data_dir / "text_features.npz", csr_matrix(np.random.rand(5, 100)))
        np.save(temp_data_dir / "type_features.npy", np.random.rand(5, 3))
        np.save(temp_data_dir / "language_features.npy", np.random.rand(5, 4))

        with pytest.raises(FileNotFoundError) as exc_info:
            load_features(temp_data_dir)

        assert "platform_features.npy" in str(exc_info.value)

    def test_load_features_all_files_missing(self, temp_data_dir):
        """Test error when all files are missing."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_features(temp_data_dir)

        assert "genre_features.npy" in str(exc_info.value)


class TestComputeSimilarities:
    """Tests for compute_similarities function."""

    def test_compute_similarities_success(self):
        """Test successful similarity computation."""
        features = {
            "genre_features": np.random.rand(5, 10),
            "text_features": csr_matrix(np.random.rand(5, 100)),
            "platform_features": np.random.rand(5, 5),
            "type_features": np.random.rand(5, 3),
            "language_features": np.random.rand(5, 4),
        }

        with patch("scripts.compute_similarities.SimilarityComputer") as MockComputer:
            mock_computer = MockComputer.return_value

            expected_similarities = {
                "genre_similarity": np.random.rand(5, 5),
                "text_similarity": np.random.rand(5, 5),
                "metadata_similarity": np.random.rand(5, 5),
                "hybrid_similarity": np.random.rand(5, 5),
            }
            mock_computer.compute_all_similarities.return_value = expected_similarities

            # Mock get_similarity_statistics
            mock_computer.get_similarity_statistics.return_value = {
                "mean": 0.5,
                "std": 0.2,
                "min": 0.0,
                "max": 1.0,
                "median": 0.5,
            }

            result = compute_similarities(
                features=features, genre_weight=0.4, text_weight=0.5, metadata_weight=0.1
            )

            # Verify computer was initialized with correct weights
            MockComputer.assert_called_once_with(
                genre_weight=0.4, text_weight=0.5, metadata_weight=0.1
            )

            # Verify compute_all_similarities was called
            mock_computer.compute_all_similarities.assert_called_once_with(features)

            # Verify statistics were computed for all similarity types
            assert mock_computer.get_similarity_statistics.call_count == 4

            # Verify result
            assert result == expected_similarities

    def test_compute_similarities_with_default_weights(self):
        """Test similarity computation with default weights."""
        features = {
            "genre_features": np.random.rand(3, 5),
            "text_features": csr_matrix(np.random.rand(3, 50)),
            "platform_features": np.random.rand(3, 3),
            "type_features": np.random.rand(3, 2),
            "language_features": np.random.rand(3, 2),
        }

        with patch("scripts.compute_similarities.SimilarityComputer") as MockComputer:
            mock_computer = MockComputer.return_value
            mock_computer.compute_all_similarities.return_value = {
                "genre_similarity": np.eye(3),
                "text_similarity": np.eye(3),
                "metadata_similarity": np.eye(3),
                "hybrid_similarity": np.eye(3),
            }
            mock_computer.get_similarity_statistics.return_value = {
                "mean": 0.33,
                "std": 0.47,
                "min": 0.0,
                "max": 1.0,
                "median": 0.0,
            }

            compute_similarities(features=features)

            # Verify default weights
            MockComputer.assert_called_once_with(
                genre_weight=0.4, text_weight=0.5, metadata_weight=0.1
            )

    def test_compute_similarities_custom_weights(self):
        """Test similarity computation with custom weights."""
        features = {
            "genre_features": np.random.rand(3, 5),
            "text_features": csr_matrix(np.random.rand(3, 50)),
            "platform_features": np.random.rand(3, 3),
            "type_features": np.random.rand(3, 2),
            "language_features": np.random.rand(3, 2),
        }

        with patch("scripts.compute_similarities.SimilarityComputer") as MockComputer:
            mock_computer = MockComputer.return_value
            mock_computer.compute_all_similarities.return_value = {
                "genre_similarity": np.eye(3),
                "text_similarity": np.eye(3),
                "metadata_similarity": np.eye(3),
                "hybrid_similarity": np.eye(3),
            }
            mock_computer.get_similarity_statistics.return_value = {
                "mean": 0.33,
                "std": 0.47,
                "min": 0.0,
                "max": 1.0,
                "median": 0.0,
            }

            compute_similarities(
                features=features, genre_weight=0.3, text_weight=0.6, metadata_weight=0.1
            )

            MockComputer.assert_called_once_with(
                genre_weight=0.3, text_weight=0.6, metadata_weight=0.1
            )


class TestSaveSimilarities:
    """Tests for save_similarities function."""

    def test_save_similarities_skip_by_default(self, temp_data_dir, capsys):
        """Test that similarities are not saved by default (save_to_disk=False)."""
        similarities = {
            "genre_similarity": np.random.rand(5, 5),
            "text_similarity": np.random.rand(5, 5),
            "metadata_similarity": np.random.rand(5, 5),
            "hybrid_similarity": np.random.rand(5, 5),
        }

        output_dir = temp_data_dir / "output"

        save_similarities(similarities, output_dir, save_to_disk=False)

        # Verify no files were created
        if output_dir.exists():
            assert len(list(output_dir.glob("*.npy"))) == 0

    def test_save_similarities_when_enabled(self, temp_data_dir):
        """Test saving similarities when explicitly enabled."""
        similarities = {
            "genre_similarity": np.random.rand(5, 5),
            "text_similarity": np.random.rand(5, 5),
            "metadata_similarity": np.random.rand(5, 5),
            "hybrid_similarity": np.random.rand(5, 5),
        }

        output_dir = temp_data_dir / "output"

        save_similarities(similarities, output_dir, save_to_disk=True)

        # Verify directory was created
        assert output_dir.exists()

        # Verify all files were saved
        assert (output_dir / "genre_similarity.npy").exists()
        assert (output_dir / "text_similarity.npy").exists()
        assert (output_dir / "metadata_similarity.npy").exists()
        assert (output_dir / "hybrid_similarity.npy").exists()

        # Verify data integrity
        loaded = np.load(output_dir / "genre_similarity.npy")
        assert loaded.shape == (5, 5)
        assert np.allclose(loaded, similarities["genre_similarity"])

    def test_save_similarities_creates_nested_directories(self, temp_data_dir):
        """Test creation of nested directories when saving."""
        similarities = {
            "genre_similarity": np.random.rand(3, 3),
            "text_similarity": np.random.rand(3, 3),
            "metadata_similarity": np.random.rand(3, 3),
            "hybrid_similarity": np.random.rand(3, 3),
        }

        output_dir = temp_data_dir / "level1" / "level2" / "sims"

        save_similarities(similarities, output_dir, save_to_disk=True)

        assert output_dir.exists()
        assert (output_dir / "genre_similarity.npy").exists()

    def test_save_similarities_overwrites_existing(self, temp_data_dir):
        """Test overwriting existing similarity files."""
        output_dir = temp_data_dir / "output"

        # Save initial similarities
        sims1 = {
            "genre_similarity": np.ones((3, 3)),
            "text_similarity": np.ones((3, 3)),
            "metadata_similarity": np.ones((3, 3)),
            "hybrid_similarity": np.ones((3, 3)),
        }
        save_similarities(sims1, output_dir, save_to_disk=True)

        # Save different similarities
        sims2 = {
            "genre_similarity": np.zeros((3, 3)),
            "text_similarity": np.zeros((3, 3)),
            "metadata_similarity": np.zeros((3, 3)),
            "hybrid_similarity": np.zeros((3, 3)),
        }
        save_similarities(sims2, output_dir, save_to_disk=True)

        # Verify new data
        loaded = np.load(output_dir / "genre_similarity.npy")
        assert np.allclose(loaded, np.zeros((3, 3)))


class TestMain:
    """Tests for main function."""

    @patch("scripts.compute_similarities.save_similarities")
    @patch("scripts.compute_similarities.compute_similarities")
    @patch("scripts.compute_similarities.load_features")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_success_with_defaults(self, mock_parse_args, mock_load, mock_compute, mock_save):
        """Test main function with default arguments."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.genre_weight = 0.4
        mock_args.text_weight = 0.5
        mock_args.metadata_weight = 0.1
        mock_args.save_similarities = False
        mock_parse_args.return_value = mock_args

        mock_features = {
            "genre_features": np.random.rand(5, 10),
            "text_features": csr_matrix(np.random.rand(5, 100)),
            "platform_features": np.random.rand(5, 5),
            "type_features": np.random.rand(5, 3),
            "language_features": np.random.rand(5, 4),
        }
        mock_load.return_value = mock_features

        mock_sims = {
            "genre_similarity": np.random.rand(5, 5),
            "text_similarity": np.random.rand(5, 5),
            "metadata_similarity": np.random.rand(5, 5),
            "hybrid_similarity": np.random.rand(5, 5),
        }
        mock_compute.return_value = mock_sims

        result = main()

        # Verify functions were called
        mock_load.assert_called_once()
        mock_compute.assert_called_once_with(
            features=mock_features, genre_weight=0.4, text_weight=0.5, metadata_weight=0.1
        )
        # Verify save was called (don't check exact path due to project_root variations)
        assert mock_save.called
        assert mock_save.call_args[0][0] == mock_sims
        assert mock_save.call_args[1]["save_to_disk"] is False

        # Verify result
        assert result == mock_sims

    @patch("scripts.compute_similarities.save_similarities")
    @patch("scripts.compute_similarities.compute_similarities")
    @patch("scripts.compute_similarities.load_features")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_with_save_enabled(self, mock_parse_args, mock_load, mock_compute, mock_save):
        """Test main function with save_similarities enabled."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.genre_weight = 0.4
        mock_args.text_weight = 0.5
        mock_args.metadata_weight = 0.1
        mock_args.save_similarities = True
        mock_parse_args.return_value = mock_args

        mock_load.return_value = {
            "genre_features": np.random.rand(3, 5),
            "text_features": csr_matrix(np.random.rand(3, 50)),
            "platform_features": np.random.rand(3, 3),
            "type_features": np.random.rand(3, 2),
            "language_features": np.random.rand(3, 2),
        }
        mock_compute.return_value = {
            "genre_similarity": np.eye(3),
            "text_similarity": np.eye(3),
            "metadata_similarity": np.eye(3),
            "hybrid_similarity": np.eye(3),
        }

        main()

        # Verify save was called with True
        call_args = mock_save.call_args
        assert call_args[1]["save_to_disk"] is True

    @patch("scripts.compute_similarities.save_similarities")
    @patch("scripts.compute_similarities.compute_similarities")
    @patch("scripts.compute_similarities.load_features")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_with_custom_weights(self, mock_parse_args, mock_load, mock_compute, mock_save):
        """Test main function with custom weights."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.genre_weight = 0.3
        mock_args.text_weight = 0.6
        mock_args.metadata_weight = 0.1
        mock_args.save_similarities = False
        mock_parse_args.return_value = mock_args

        mock_load.return_value = {
            "genre_features": np.random.rand(3, 5),
            "text_features": csr_matrix(np.random.rand(3, 50)),
            "platform_features": np.random.rand(3, 3),
            "type_features": np.random.rand(3, 2),
            "language_features": np.random.rand(3, 2),
        }
        mock_compute.return_value = {
            "genre_similarity": np.eye(3),
            "text_similarity": np.eye(3),
            "metadata_similarity": np.eye(3),
            "hybrid_similarity": np.eye(3),
        }

        main()

        mock_compute.assert_called_once_with(
            features=mock_load.return_value, genre_weight=0.3, text_weight=0.6, metadata_weight=0.1
        )

    @patch("scripts.compute_similarities.load_features")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("sys.exit")
    def test_main_validates_weights_sum(self, mock_exit, mock_parse_args, mock_load):
        """Test main function validates that weights sum > 0."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.genre_weight = 0.0
        mock_args.text_weight = 0.0
        mock_args.metadata_weight = 0.0
        mock_args.save_similarities = False
        mock_parse_args.return_value = mock_args

        main()

        # Should exit with error (may be called multiple times due to exception handling)
        assert mock_exit.called
        assert 1 in [call[0][0] for call in mock_exit.call_args_list]

    @patch("scripts.compute_similarities.load_features")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("sys.exit")
    def test_main_handles_load_error(self, mock_exit, mock_parse_args, mock_load):
        """Test main function handles loading errors."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.genre_weight = 0.4
        mock_args.text_weight = 0.5
        mock_args.metadata_weight = 0.1
        mock_args.save_similarities = False
        mock_parse_args.return_value = mock_args

        mock_load.side_effect = FileNotFoundError("Features not found")

        main()

        mock_exit.assert_called_once_with(1)

    @patch("scripts.compute_similarities.compute_similarities")
    @patch("scripts.compute_similarities.load_features")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("sys.exit")
    def test_main_handles_compute_error(self, mock_exit, mock_parse_args, mock_load, mock_compute):
        """Test main function handles computation errors."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.genre_weight = 0.4
        mock_args.text_weight = 0.5
        mock_args.metadata_weight = 0.1
        mock_args.save_similarities = False
        mock_parse_args.return_value = mock_args

        mock_load.return_value = {
            "genre_features": np.random.rand(3, 5),
            "text_features": csr_matrix(np.random.rand(3, 50)),
            "platform_features": np.random.rand(3, 3),
            "type_features": np.random.rand(3, 2),
            "language_features": np.random.rand(3, 2),
        }
        mock_compute.side_effect = Exception("Computation failed")

        main()

        mock_exit.assert_called_once_with(1)

    @patch("scripts.compute_similarities.save_similarities")
    @patch("scripts.compute_similarities.compute_similarities")
    @patch("scripts.compute_similarities.load_features")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("sys.exit")
    def test_main_handles_save_error(
        self, mock_exit, mock_parse_args, mock_load, mock_compute, mock_save
    ):
        """Test main function handles save errors."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.genre_weight = 0.4
        mock_args.text_weight = 0.5
        mock_args.metadata_weight = 0.1
        mock_args.save_similarities = True
        mock_parse_args.return_value = mock_args

        mock_load.return_value = {
            "genre_features": np.random.rand(3, 5),
            "text_features": csr_matrix(np.random.rand(3, 50)),
            "platform_features": np.random.rand(3, 3),
            "type_features": np.random.rand(3, 2),
            "language_features": np.random.rand(3, 2),
        }
        mock_compute.return_value = {
            "genre_similarity": np.eye(3),
            "text_similarity": np.eye(3),
            "metadata_similarity": np.eye(3),
            "hybrid_similarity": np.eye(3),
        }
        mock_save.side_effect = Exception("Save failed")

        main()

        mock_exit.assert_called_once_with(1)
