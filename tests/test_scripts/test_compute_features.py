"""
Tests for scripts/compute_features.py
"""

import pickle
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

# Import functions from the script
from scripts.compute_features import extract_features, load_prepared_data, main, save_features


class TestLoadPreparedData:
    """Tests for load_prepared_data function."""

    def test_load_prepared_data_success(self, temp_data_dir, sample_shows_df):
        """Test successful loading of prepared data."""
        # Create metadata CSV
        metadata_path = temp_data_dir / "shows_metadata.csv"
        sample_shows_df.to_csv(metadata_path, index=False)

        result = load_prepared_data(temp_data_dir)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_shows_df)
        assert list(result.columns) == list(sample_shows_df.columns)

    def test_load_prepared_data_file_not_found(self, temp_data_dir):
        """Test error when metadata file doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_prepared_data(temp_data_dir)

        assert "shows_metadata.csv" in str(exc_info.value)
        assert "Run fetch_and_prepare_data.py first" in str(exc_info.value)

    def test_load_prepared_data_with_various_dtypes(self, temp_data_dir):
        """Test loading data with various data types."""
        # Create test data with different types
        test_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Show1", "Show2", "Show3"],
                "genres": [["Drama"], ["Comedy"], ["Action", "Sci-Fi"]],
                "rating_avg": [8.5, 7.2, None],
                "platform": ["Netflix", None, "HBO"],
            }
        )

        metadata_path = temp_data_dir / "shows_metadata.csv"
        test_df.to_csv(metadata_path, index=False)

        result = load_prepared_data(temp_data_dir)

        assert len(result) == 3
        assert result["id"].dtype == np.int64


class TestExtractFeatures:
    """Tests for extract_features function."""

    def test_extract_features_success(self, sample_shows_df):
        """Test successful feature extraction."""
        with patch("scripts.compute_features.FeatureExtractor") as MockExtractor:
            mock_extractor = MockExtractor.return_value

            # Mock return values
            expected_features = {
                "genre_features": np.random.rand(3, 10),
                "text_features": csr_matrix(np.random.rand(3, 100)),
                "platform_features": np.random.rand(3, 5),
                "type_features": np.random.rand(3, 2),
                "language_features": np.random.rand(3, 3),
                "tfidf_vectorizer": Mock(),
                "genre_encoder": Mock(),
            }
            mock_extractor.extract_all_features.return_value = expected_features

            result = extract_features(
                df=sample_shows_df,
                max_text_features=500,
                text_min_df=2,
                text_max_df=0.8,
                top_n_platforms=20,
                top_n_languages=5,
            )

            # Verify extractor was initialized with correct params
            MockExtractor.assert_called_once_with(
                max_text_features=500,
                text_min_df=2,
                text_max_df=0.8,
                top_n_platforms=20,
                top_n_languages=5,
            )

            # Verify extract_all_features was called
            mock_extractor.extract_all_features.assert_called_once_with(sample_shows_df)

            # Verify result
            assert result == expected_features

    def test_extract_features_with_defaults(self, sample_shows_df):
        """Test feature extraction with default parameters."""
        with patch("scripts.compute_features.FeatureExtractor") as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_extractor.extract_all_features.return_value = {
                "genre_features": np.array([]),
                "text_features": csr_matrix((0, 0)),
                "platform_features": np.array([]),
                "type_features": np.array([]),
                "language_features": np.array([]),
                "tfidf_vectorizer": Mock(),
                "genre_encoder": Mock(),
            }

            extract_features(df=sample_shows_df)

            # Verify default params were used
            MockExtractor.assert_called_once_with(
                max_text_features=500,
                text_min_df=2,
                text_max_df=0.8,
                top_n_platforms=20,
                top_n_languages=5,
            )

    def test_extract_features_custom_params(self, sample_shows_df):
        """Test feature extraction with custom parameters."""
        with patch("scripts.compute_features.FeatureExtractor") as MockExtractor:
            mock_extractor = MockExtractor.return_value
            mock_extractor.extract_all_features.return_value = {
                "genre_features": np.array([]),
                "text_features": csr_matrix((0, 0)),
                "platform_features": np.array([]),
                "type_features": np.array([]),
                "language_features": np.array([]),
                "tfidf_vectorizer": Mock(),
                "genre_encoder": Mock(),
            }

            extract_features(
                df=sample_shows_df,
                max_text_features=1000,
                text_min_df=5,
                text_max_df=0.9,
                top_n_platforms=50,
                top_n_languages=10,
            )

            MockExtractor.assert_called_once_with(
                max_text_features=1000,
                text_min_df=5,
                text_max_df=0.9,
                top_n_platforms=50,
                top_n_languages=10,
            )


class MockSerializable:
    """Simple mock object that can be pickled."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestSaveFeatures:
    """Tests for save_features function."""

    def test_save_features_success(self, temp_data_dir):
        """Test successful saving of features."""
        output_dir = temp_data_dir / "features"

        # Create simple mock objects that can be pickled
        mock_vectorizer = MockSerializable()
        mock_encoder = MockSerializable()

        # Create sample features
        features = {
            "genre_features": np.random.rand(3, 10),
            "text_features": csr_matrix(np.random.rand(3, 100)),
            "platform_features": np.random.rand(3, 5),
            "type_features": np.random.rand(3, 2),
            "language_features": np.random.rand(3, 3),
            "tfidf_vectorizer": mock_vectorizer,
            "genre_encoder": mock_encoder,
        }

        save_features(features, output_dir)

        # Verify directory was created
        assert output_dir.exists()

        # Verify all files were created
        assert (output_dir / "genre_features.npy").exists()
        assert (output_dir / "text_features.npz").exists()
        assert (output_dir / "platform_features.npy").exists()
        assert (output_dir / "type_features.npy").exists()
        assert (output_dir / "language_features.npy").exists()
        assert (output_dir / "tfidf_vectorizer.pkl").exists()
        assert (output_dir / "genre_encoder.pkl").exists()

        # Verify data can be loaded back
        loaded_genre = np.load(output_dir / "genre_features.npy")
        assert loaded_genre.shape == features["genre_features"].shape

    def test_save_features_creates_nested_directories(self, temp_data_dir):
        """Test creation of nested directories."""
        output_dir = temp_data_dir / "level1" / "level2" / "features"

        mock_vectorizer = MockSerializable()
        mock_encoder = MockSerializable()

        features = {
            "genre_features": np.random.rand(2, 5),
            "text_features": csr_matrix(np.random.rand(2, 50)),
            "platform_features": np.random.rand(2, 3),
            "type_features": np.random.rand(2, 2),
            "language_features": np.random.rand(2, 2),
            "tfidf_vectorizer": mock_vectorizer,
            "genre_encoder": mock_encoder,
        }

        save_features(features, output_dir)

        assert output_dir.exists()
        assert (output_dir / "genre_features.npy").exists()

    def test_save_features_overwrites_existing(self, temp_data_dir):
        """Test overwriting existing files."""
        output_dir = temp_data_dir / "features"

        mock_vectorizer1 = MockSerializable()
        mock_encoder1 = MockSerializable()
        mock_vectorizer2 = MockSerializable()
        mock_encoder2 = MockSerializable()

        # Save initial features
        features1 = {
            "genre_features": np.array([[1, 2, 3]]),
            "text_features": csr_matrix(np.array([[1, 2, 3]])),
            "platform_features": np.array([[1]]),
            "type_features": np.array([[1]]),
            "language_features": np.array([[1]]),
            "tfidf_vectorizer": mock_vectorizer1,
            "genre_encoder": mock_encoder1,
        }
        save_features(features1, output_dir)

        # Save different features
        features2 = {
            "genre_features": np.array([[4, 5, 6, 7]]),
            "text_features": csr_matrix(np.array([[4, 5, 6, 7]])),
            "platform_features": np.array([[2]]),
            "type_features": np.array([[2]]),
            "language_features": np.array([[2]]),
            "tfidf_vectorizer": mock_vectorizer2,
            "genre_encoder": mock_encoder2,
        }
        save_features(features2, output_dir)

        # Verify new data was saved
        loaded_genre = np.load(output_dir / "genre_features.npy")
        assert loaded_genre.shape == (1, 4)
        assert np.array_equal(loaded_genre, features2["genre_features"])

    def test_save_features_pickle_serialization(self, temp_data_dir):
        """Test pickle serialization of encoders."""
        output_dir = temp_data_dir / "features"

        # Create simple serializable objects
        mock_vectorizer = MockSerializable(test_attr="test_value")
        mock_encoder = MockSerializable(classes_=["Drama", "Comedy", "Action"])

        features = {
            "genre_features": np.random.rand(2, 3),
            "text_features": csr_matrix(np.random.rand(2, 10)),
            "platform_features": np.random.rand(2, 2),
            "type_features": np.random.rand(2, 1),
            "language_features": np.random.rand(2, 1),
            "tfidf_vectorizer": mock_vectorizer,
            "genre_encoder": mock_encoder,
        }

        save_features(features, output_dir)

        # Load and verify pickled objects
        with open(output_dir / "tfidf_vectorizer.pkl", "rb") as f:
            loaded_vectorizer = pickle.load(f)
        with open(output_dir / "genre_encoder.pkl", "rb") as f:
            loaded_encoder = pickle.load(f)

        assert loaded_vectorizer.test_attr == "test_value"
        assert loaded_encoder.classes_ == ["Drama", "Comedy", "Action"]


class TestMain:
    """Tests for main function."""

    @patch("scripts.compute_features.save_features")
    @patch("scripts.compute_features.extract_features")
    @patch("scripts.compute_features.load_prepared_data")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_success_with_defaults(
        self, mock_parse_args, mock_load, mock_extract, mock_save, sample_shows_df
    ):
        """Test main function with default arguments."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.max_text_features = 500
        mock_args.text_min_df = 2
        mock_args.text_max_df = 0.8
        mock_args.top_n_platforms = 20
        mock_args.top_n_languages = 5
        mock_parse_args.return_value = mock_args

        mock_load.return_value = sample_shows_df

        mock_features = {
            "genre_features": np.random.rand(3, 10),
            "text_features": csr_matrix(np.random.rand(3, 100)),
            "platform_features": np.random.rand(3, 5),
            "type_features": np.random.rand(3, 2),
            "language_features": np.random.rand(3, 3),
            "tfidf_vectorizer": Mock(),
            "genre_encoder": Mock(),
        }
        mock_extract.return_value = mock_features

        main()

        # Verify functions were called
        mock_load.assert_called_once()
        mock_extract.assert_called_once_with(
            df=sample_shows_df,
            max_text_features=500,
            text_min_df=2,
            text_max_df=0.8,
            top_n_platforms=20,
            top_n_languages=5,
        )
        mock_save.assert_called_once()

    @patch("scripts.compute_features.save_features")
    @patch("scripts.compute_features.extract_features")
    @patch("scripts.compute_features.load_prepared_data")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_success_with_custom_args(
        self, mock_parse_args, mock_load, mock_extract, mock_save, sample_shows_df
    ):
        """Test main function with custom arguments."""
        mock_args = Mock()
        mock_args.input_dir = "custom/input"
        mock_args.output_dir = "custom/output"
        mock_args.max_text_features = 1000
        mock_args.text_min_df = 5
        mock_args.text_max_df = 0.9
        mock_args.top_n_platforms = 50
        mock_args.top_n_languages = 10
        mock_parse_args.return_value = mock_args

        mock_load.return_value = sample_shows_df
        mock_extract.return_value = {
            "genre_features": np.array([]),
            "text_features": csr_matrix((0, 0)),
            "platform_features": np.array([]),
            "type_features": np.array([]),
            "language_features": np.array([]),
            "tfidf_vectorizer": Mock(),
            "genre_encoder": Mock(),
        }

        main()

        mock_extract.assert_called_once_with(
            df=sample_shows_df,
            max_text_features=1000,
            text_min_df=5,
            text_max_df=0.9,
            top_n_platforms=50,
            top_n_languages=10,
        )

    @patch("scripts.compute_features.load_prepared_data")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("sys.exit")
    def test_main_handles_load_error(self, mock_exit, mock_parse_args, mock_load):
        """Test main function handles loading errors."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.max_text_features = 500
        mock_args.text_min_df = 2
        mock_args.text_max_df = 0.8
        mock_args.top_n_platforms = 20
        mock_args.top_n_languages = 5
        mock_parse_args.return_value = mock_args

        mock_load.side_effect = FileNotFoundError("File not found")

        main()

        mock_exit.assert_called_once_with(1)

    @patch("scripts.compute_features.extract_features")
    @patch("scripts.compute_features.load_prepared_data")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("sys.exit")
    def test_main_handles_extract_error(
        self, mock_exit, mock_parse_args, mock_load, mock_extract, sample_shows_df
    ):
        """Test main function handles extraction errors."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.max_text_features = 500
        mock_args.text_min_df = 2
        mock_args.text_max_df = 0.8
        mock_args.top_n_platforms = 20
        mock_args.top_n_languages = 5
        mock_parse_args.return_value = mock_args

        mock_load.return_value = sample_shows_df
        mock_extract.side_effect = Exception("Extraction failed")

        main()

        mock_exit.assert_called_once_with(1)

    @patch("scripts.compute_features.save_features")
    @patch("scripts.compute_features.extract_features")
    @patch("scripts.compute_features.load_prepared_data")
    @patch("argparse.ArgumentParser.parse_args")
    @patch("sys.exit")
    def test_main_handles_save_error(
        self, mock_exit, mock_parse_args, mock_load, mock_extract, mock_save, sample_shows_df
    ):
        """Test main function handles save errors."""
        mock_args = Mock()
        mock_args.input_dir = "data/processed"
        mock_args.output_dir = "data/processed"
        mock_args.max_text_features = 500
        mock_args.text_min_df = 2
        mock_args.text_max_df = 0.8
        mock_args.top_n_platforms = 20
        mock_args.top_n_languages = 5
        mock_parse_args.return_value = mock_args

        mock_load.return_value = sample_shows_df
        mock_extract.return_value = {
            "genre_features": np.array([]),
            "text_features": csr_matrix((0, 0)),
            "platform_features": np.array([]),
            "type_features": np.array([]),
            "language_features": np.array([]),
            "tfidf_vectorizer": Mock(),
            "genre_encoder": Mock(),
        }
        mock_save.side_effect = Exception("Save failed")

        main()

        mock_exit.assert_called_once_with(1)

    @patch("scripts.compute_features.save_features")
    @patch("scripts.compute_features.extract_features")
    @patch("scripts.compute_features.load_prepared_data")
    @patch("argparse.ArgumentParser.parse_args")
    def test_main_completes_full_workflow(
        self, mock_parse_args, mock_load, mock_extract, mock_save, sample_shows_df, temp_data_dir
    ):
        """Test main function completes full workflow."""
        mock_args = Mock()
        mock_args.input_dir = str(temp_data_dir)
        mock_args.output_dir = str(temp_data_dir / "output")
        mock_args.max_text_features = 100
        mock_args.text_min_df = 1
        mock_args.text_max_df = 0.95
        mock_args.top_n_platforms = 10
        mock_args.top_n_languages = 3
        mock_parse_args.return_value = mock_args

        mock_load.return_value = sample_shows_df

        features = {
            "genre_features": np.random.rand(len(sample_shows_df), 10),
            "text_features": csr_matrix(np.random.rand(len(sample_shows_df), 100)),
            "platform_features": np.random.rand(len(sample_shows_df), 5),
            "type_features": np.random.rand(len(sample_shows_df), 2),
            "language_features": np.random.rand(len(sample_shows_df), 3),
            "tfidf_vectorizer": Mock(),
            "genre_encoder": Mock(),
        }
        mock_extract.return_value = features

        main()

        # Verify all steps executed
        assert mock_load.call_count == 1
        assert mock_extract.call_count == 1
        assert mock_save.call_count == 1
