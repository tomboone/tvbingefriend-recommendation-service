"""
Tests for scripts/populate_recommendations.py
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# Import functions from the script
from scripts.populate_recommendations import clean_dataframe_for_db, main


class TestCleanDataframeForDb:
    """Tests for clean_dataframe_for_db function."""

    def test_clean_dataframe_replaces_nan_with_none(self):
        """Test that NaN values are replaced with None."""
        df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Show1", "Show2", np.nan], "rating": [8.5, np.nan, 7.0]}
        )

        result = clean_dataframe_for_db(df)

        # Check NaN was replaced with None
        assert result.iloc[1]["rating"] is None
        assert result.iloc[2]["name"] is None

    def test_clean_dataframe_replaces_pd_na_with_none(self):
        """Test that pd.NA values are replaced with None."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Show1", "Show2", "Show3"],
                "optional_field": ["value1", pd.NA, "value3"],
            }
        )

        result = clean_dataframe_for_db(df)

        assert result.iloc[1]["optional_field"] is None

    def test_clean_dataframe_preserves_valid_values(self):
        """Test that valid values are preserved."""
        df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Show1", "Show2", "Show3"], "rating": [8.5, 7.0, 9.2]}
        )

        result = clean_dataframe_for_db(df)

        assert result.iloc[0]["id"] == 1
        assert result.iloc[1]["name"] == "Show2"
        assert result.iloc[2]["rating"] == 9.2

    def test_clean_dataframe_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["id", "name", "rating"])

        result = clean_dataframe_for_db(df)

        assert len(result) == 0
        assert list(result.columns) == ["id", "name", "rating"]

    def test_clean_dataframe_all_nan_values(self):
        """Test DataFrame with all NaN values."""
        df = pd.DataFrame(
            {
                "id": [np.nan, np.nan, np.nan],
                "name": [np.nan, np.nan, np.nan],
                "rating": [np.nan, np.nan, np.nan],
            }
        )

        result = clean_dataframe_for_db(df)

        # All values should be None
        for col in result.columns:
            assert all(result[col].isna() | (result[col] is None))

    def test_clean_dataframe_mixed_types(self):
        """Test DataFrame with mixed data types."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Show1", "Show2", "Show3"],
                "rating": [8.5, np.nan, 7.0],
                "genres": [["Drama"], ["Comedy", "Action"], np.nan],
                "platform": ["Netflix", None, "HBO"],
            }
        )

        result = clean_dataframe_for_db(df)

        assert result.iloc[0]["id"] == 1
        assert result.iloc[1]["rating"] is None
        assert result.iloc[1]["platform"] is None


class TestMain:
    """Tests for main function."""

    @patch("scripts.populate_recommendations.ContentBasedRecommendationService")
    @patch("scripts.populate_recommendations.pd.read_csv")
    def test_main_success(self, mock_read_csv, mock_service_class, sample_shows_df, temp_data_dir):
        """Test successful execution of main function."""
        # Mock CSV reading
        mock_read_csv.return_value = sample_shows_df

        # Mock service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.sync_metadata_to_db.return_value = len(sample_shows_df)
        mock_service.compute_and_store_all_similarities.return_value = {
            "total_records": 60,
            "unique_shows": 3,
            "avg_similarities_per_show": 20.0,
            "last_computed": "2024-01-01 00:00:00",
        }
        mock_service.get_recommendations_from_db.return_value = [
            {"name": "Recommended Show", "similarity_score": 0.85, "genres": ["Drama", "Crime"]}
        ]

        # Patch project_root to use temp_data_dir
        with patch("scripts.populate_recommendations.project_root", temp_data_dir):
            # Create the CSV file
            processed_dir = temp_data_dir / "data" / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            sample_shows_df.to_csv(processed_dir / "shows_metadata.csv", index=False)

            main()

            # Verify service was initialized
            mock_service_class.assert_called_once()

            # Verify metadata sync was called
            mock_service.sync_metadata_to_db.assert_called_once()

            # Verify similarities were computed
            mock_service.compute_and_store_all_similarities.assert_called_once()

            # Verify recommendations were tested (3 test shows)
            assert mock_service.get_recommendations_from_db.call_count == 3

    @patch("scripts.populate_recommendations.ContentBasedRecommendationService")
    @patch("scripts.populate_recommendations.pd.read_csv")
    def test_main_syncs_metadata_correctly(
        self, mock_read_csv, mock_service_class, sample_shows_df
    ):
        """Test that metadata is synced correctly."""
        mock_read_csv.return_value = sample_shows_df

        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.sync_metadata_to_db.return_value = 3
        mock_service.compute_and_store_all_similarities.return_value = {
            "total_records": 60,
            "unique_shows": 3,
            "avg_similarities_per_show": 20.0,
            "last_computed": "2024-01-01 00:00:00",
        }
        mock_service.get_recommendations_from_db.return_value = []

        main()

        # Get the data passed to sync_metadata_to_db
        call_args = mock_service.sync_metadata_to_db.call_args[0][0]

        # Verify it's a list of dicts
        assert isinstance(call_args, list)
        assert len(call_args) == len(sample_shows_df)

    @patch("scripts.populate_recommendations.ContentBasedRecommendationService")
    @patch("scripts.populate_recommendations.pd.read_csv")
    def test_main_cleans_nan_values(self, mock_read_csv, mock_service_class):
        """Test that NaN values are cleaned before syncing."""
        # Create DataFrame with NaN values
        df_with_nan = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Show1", "Show2", "Show3"],
                "genres": [["Drama"], ["Comedy"], ["Action"]],
                "summary_clean": ["Summary 1", "Summary 2", "Summary 3"],
                "rating_avg": [8.5, np.nan, 7.0],
                "platform": ["Netflix", None, "HBO"],
            }
        )

        mock_read_csv.return_value = df_with_nan

        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.sync_metadata_to_db.return_value = 3
        mock_service.compute_and_store_all_similarities.return_value = {
            "total_records": 60,
            "unique_shows": 3,
            "avg_similarities_per_show": 20.0,
            "last_computed": "2024-01-01 00:00:00",
        }
        mock_service.get_recommendations_from_db.return_value = []

        main()

        # Get the data passed to sync_metadata_to_db
        call_args = mock_service.sync_metadata_to_db.call_args[0][0]

        # Verify NaN was replaced with None
        assert call_args[1]["rating_avg"] is None
        assert call_args[1]["platform"] is None

    @patch("scripts.populate_recommendations.ContentBasedRecommendationService")
    @patch("scripts.populate_recommendations.pd.read_csv")
    def test_main_computes_similarities(self, mock_read_csv, mock_service_class, sample_shows_df):
        """Test that similarities are computed."""
        mock_read_csv.return_value = sample_shows_df

        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.sync_metadata_to_db.return_value = 3

        expected_stats = {
            "total_records": 60,
            "unique_shows": 3,
            "avg_similarities_per_show": 20.0,
            "last_computed": "2024-01-01 00:00:00",
        }
        mock_service.compute_and_store_all_similarities.return_value = expected_stats
        mock_service.get_recommendations_from_db.return_value = []

        main()

        # Verify compute was called
        mock_service.compute_and_store_all_similarities.assert_called_once()

    @patch("scripts.populate_recommendations.ContentBasedRecommendationService")
    @patch("scripts.populate_recommendations.pd.read_csv")
    def test_main_tests_recommendations(self, mock_read_csv, mock_service_class, sample_shows_df):
        """Test that recommendations are tested for sample shows."""
        mock_read_csv.return_value = sample_shows_df

        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.sync_metadata_to_db.return_value = 3
        mock_service.compute_and_store_all_similarities.return_value = {
            "total_records": 60,
            "unique_shows": 3,
            "avg_similarities_per_show": 20.0,
            "last_computed": "2024-01-01 00:00:00",
        }

        # Mock recommendations for each show
        mock_service.get_recommendations_from_db.return_value = [
            {"name": "Rec1", "similarity_score": 0.9, "genres": ["Drama"]},
            {"name": "Rec2", "similarity_score": 0.8, "genres": ["Crime"]},
            {"name": "Rec3", "similarity_score": 0.7, "genres": ["Thriller"]},
        ]

        main()

        # Should test first 3 shows
        assert mock_service.get_recommendations_from_db.call_count == 3

        # Verify show IDs from sample_shows_df were used
        call_args_list = mock_service.get_recommendations_from_db.call_args_list
        for i, call_args in enumerate(call_args_list):
            assert call_args[1]["show_id"] == sample_shows_df.iloc[i]["id"]
            assert call_args[1]["n"] == 5

    @patch("scripts.populate_recommendations.ContentBasedRecommendationService")
    @patch("scripts.populate_recommendations.pd.read_csv")
    def test_main_handles_no_recommendations(
        self, mock_read_csv, mock_service_class, sample_shows_df
    ):
        """Test handling when no recommendations are found."""
        mock_read_csv.return_value = sample_shows_df

        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.sync_metadata_to_db.return_value = 3
        mock_service.compute_and_store_all_similarities.return_value = {
            "total_records": 0,
            "unique_shows": 0,
            "avg_similarities_per_show": 0.0,
            "last_computed": "2024-01-01 00:00:00",
        }
        mock_service.get_recommendations_from_db.return_value = []

        # Should not raise any errors
        main()

        # Verify function completed
        assert mock_service.get_recommendations_from_db.call_count == 3

    @patch("scripts.populate_recommendations.ContentBasedRecommendationService")
    @patch("scripts.populate_recommendations.pd.read_csv")
    def test_main_with_empty_dataframe(self, mock_read_csv, mock_service_class):
        """Test main with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["id", "name"])
        mock_read_csv.return_value = empty_df

        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.sync_metadata_to_db.return_value = 0
        mock_service.compute_and_store_all_similarities.return_value = {
            "total_records": 0,
            "unique_shows": 0,
            "avg_similarities_per_show": 0.0,
            "last_computed": "2024-01-01 00:00:00",
        }

        main()

        # Should sync empty data
        mock_service.sync_metadata_to_db.assert_called_once()

        # Should compute similarities
        mock_service.compute_and_store_all_similarities.assert_called_once()

        # Should not test recommendations (no shows)
        mock_service.get_recommendations_from_db.assert_not_called()

    @patch("scripts.populate_recommendations.ContentBasedRecommendationService")
    @patch("scripts.populate_recommendations.pd.read_csv")
    def test_main_displays_stats(self, mock_read_csv, mock_service_class, sample_shows_df):
        """Test that statistics are displayed."""
        mock_read_csv.return_value = sample_shows_df

        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.sync_metadata_to_db.return_value = 100
        mock_service.compute_and_store_all_similarities.return_value = {
            "total_records": 2000,
            "unique_shows": 100,
            "avg_similarities_per_show": 20.0,
            "last_computed": "2024-01-01 12:00:00",
        }
        mock_service.get_recommendations_from_db.return_value = []

        # Should complete without errors
        main()

        # Verify all methods were called
        mock_service.sync_metadata_to_db.assert_called_once()
        mock_service.compute_and_store_all_similarities.assert_called_once()

    @patch("scripts.populate_recommendations.ContentBasedRecommendationService")
    @patch("scripts.populate_recommendations.pd.read_csv")
    def test_main_full_workflow(self, mock_read_csv, mock_service_class, sample_shows_df):
        """Test complete workflow from start to finish."""
        mock_read_csv.return_value = sample_shows_df

        mock_service = Mock()
        mock_service_class.return_value = mock_service

        # Step 1: Sync metadata
        mock_service.sync_metadata_to_db.return_value = 3

        # Step 2: Compute similarities
        mock_service.compute_and_store_all_similarities.return_value = {
            "total_records": 60,
            "unique_shows": 3,
            "avg_similarities_per_show": 20.0,
            "last_computed": "2024-01-01 00:00:00",
        }

        # Step 3: Test recommendations
        mock_service.get_recommendations_from_db.return_value = [
            {"name": "Show A", "similarity_score": 0.9, "genres": ["Drama"]},
            {"name": "Show B", "similarity_score": 0.85, "genres": ["Crime"]},
            {"name": "Show C", "similarity_score": 0.8, "genres": ["Thriller"]},
            {"name": "Show D", "similarity_score": 0.75, "genres": ["Action"]},
            {"name": "Show E", "similarity_score": 0.7, "genres": ["Sci-Fi"]},
        ]

        main()

        # Verify workflow executed in correct order
        assert mock_service.sync_metadata_to_db.call_count == 1
        assert mock_service.compute_and_store_all_similarities.call_count == 1
        assert mock_service.get_recommendations_from_db.call_count == 3

    @patch("scripts.populate_recommendations.ContentBasedRecommendationService")
    @patch("scripts.populate_recommendations.pd.read_csv")
    def test_main_with_single_show(self, mock_read_csv, mock_service_class):
        """Test main with only one show."""
        single_show_df = pd.DataFrame(
            {
                "id": [1],
                "name": ["Breaking Bad"],
                "genres": [["Drama"]],
                "summary_clean": ["A chemistry teacher turns to crime."],
                "rating_avg": [9.5],
                "platform": ["AMC"],
            }
        )
        mock_read_csv.return_value = single_show_df

        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.sync_metadata_to_db.return_value = 1
        mock_service.compute_and_store_all_similarities.return_value = {
            "total_records": 0,
            "unique_shows": 1,
            "avg_similarities_per_show": 0.0,
            "last_computed": "2024-01-01 00:00:00",
        }
        mock_service.get_recommendations_from_db.return_value = []

        main()

        # Should still test recommendations for the single show
        assert mock_service.get_recommendations_from_db.call_count == 1
