"""
Tests for scripts/fetch_and_prepare_data.py
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import sys
import argparse

# Import functions from the script
from scripts.fetch_and_prepare_data import (
    fetch_all_shows,
    prepare_show_data,
    save_prepared_data,
    main
)


class TestFetchAllShows:
    """Tests for fetch_all_shows function."""

    def test_fetch_all_shows_success(self, sample_shows_list):
        """Test successful fetching of shows."""
        with patch('scripts.fetch_and_prepare_data.ShowDataLoader') as MockLoader:
            mock_loader = MockLoader.return_value
            mock_loader.get_all_shows.return_value = sample_shows_list

            result = fetch_all_shows(
                show_service_url='http://test.com',
                batch_size=100,
                max_shows=10
            )

            # Verify loader was initialized correctly
            MockLoader.assert_called_once_with(show_service_url='http://test.com')

            # Verify get_all_shows was called with correct params
            mock_loader.get_all_shows.assert_called_once_with(
                batch_size=100,
                max_shows=10
            )

            # Verify result is a DataFrame
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert 'id' in result.columns
            assert 'name' in result.columns

    def test_fetch_all_shows_with_defaults(self, sample_shows_list):
        """Test fetching shows with default parameters."""
        with patch('scripts.fetch_and_prepare_data.ShowDataLoader') as MockLoader:
            mock_loader = MockLoader.return_value
            mock_loader.get_all_shows.return_value = sample_shows_list

            result = fetch_all_shows()

            # Verify loader was initialized with None URL
            MockLoader.assert_called_once_with(show_service_url=None)

            # Verify get_all_shows was called with defaults
            mock_loader.get_all_shows.assert_called_once_with(
                batch_size=1000,
                max_shows=None
            )

            assert len(result) == 3

    def test_fetch_all_shows_empty_result(self):
        """Test handling of empty results."""
        with patch('scripts.fetch_and_prepare_data.ShowDataLoader') as MockLoader:
            mock_loader = MockLoader.return_value
            mock_loader.get_all_shows.return_value = []

            result = fetch_all_shows()

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    def test_fetch_all_shows_custom_batch_size(self, sample_shows_list):
        """Test with custom batch size."""
        with patch('scripts.fetch_and_prepare_data.ShowDataLoader') as MockLoader:
            mock_loader = MockLoader.return_value
            mock_loader.get_all_shows.return_value = sample_shows_list

            result = fetch_all_shows(batch_size=500, max_shows=100)

            mock_loader.get_all_shows.assert_called_once_with(
                batch_size=500,
                max_shows=100
            )


class TestPrepareShowData:
    """Tests for prepare_show_data function."""

    def test_prepare_show_data_success(self):
        """Test successful data preparation."""
        # Create test data with nested fields
        raw_data = {
            'id': [1, 2, 3],
            'name': ['Show1', 'Show2', 'Show3'],
            'genres': [['Drama'], ['Comedy'], ['Action']],
            'summary': [
                '<p>Show 1 summary</p>',
                '<p>Show 2 summary</p>',
                '<p>Show 3 summary</p>'
            ],
            'type': ['Scripted', 'Scripted', 'Animation'],
            'language': ['English', 'English', 'English'],
            'status': ['Running', 'Ended', 'Running'],
            'rating': [
                {'average': 8.5},
                {'average': 7.0},
                {'average': None}
            ],
            'network': [
                {'name': 'NBC'},
                None,
                {'name': 'ABC'}
            ],
            'webchannel': [
                None,
                {'name': 'Netflix'},
                None
            ]
        }
        df = pd.DataFrame(raw_data)

        result = prepare_show_data(df)

        # Verify all essential columns are present
        expected_columns = [
            'id', 'name', 'genres', 'summary', 'summary_clean',
            'type', 'language', 'status', 'platform', 'rating_avg'
        ]
        for col in expected_columns:
            assert col in result.columns

        # Verify rating extraction
        assert result.iloc[0]['rating_avg'] == 8.5
        assert result.iloc[1]['rating_avg'] == 7.0
        assert pd.isna(result.iloc[2]['rating_avg'])

        # Verify platform extraction (network preferred over webchannel)
        assert result.iloc[0]['platform'] == 'NBC'
        assert result.iloc[1]['platform'] == 'Netflix'
        assert result.iloc[2]['platform'] == 'ABC'

        # Verify HTML cleaning
        assert result.iloc[0]['summary_clean'] == 'Show 1 summary'
        assert '<p>' not in result.iloc[0]['summary_clean']

    def test_prepare_show_data_handles_missing_nested_fields(self):
        """Test handling of missing nested fields."""
        raw_data = {
            'id': [1, 2],
            'name': ['Show1', 'Show2'],
            'genres': [['Drama'], []],
            'summary': ['<p>Summary</p>', None],
            'type': ['Scripted', None],
            'language': ['English', None],
            'status': ['Running', None],
            'rating': [None, None],
            'network': [None, None],
            'webchannel': [None, None]
        }
        df = pd.DataFrame(raw_data)

        result = prepare_show_data(df)

        # Verify it doesn't crash and handles None values
        assert len(result) == 2
        assert pd.isna(result.iloc[0]['rating_avg'])
        assert pd.isna(result.iloc[0]['platform'])

    def test_prepare_show_data_with_invalid_rating_structure(self):
        """Test handling of invalid rating structure."""
        raw_data = {
            'id': [1, 2, 3],
            'name': ['Show1', 'Show2', 'Show3'],
            'genres': [[], [], []],
            'summary': ['<p>S1</p>', '<p>S2</p>', '<p>S3</p>'],
            'type': ['Scripted', 'Scripted', 'Scripted'],
            'language': ['English', 'English', 'English'],
            'status': ['Running', 'Running', 'Running'],
            'rating': ['invalid', None, {}],  # Invalid structures
            'network': [None, None, None],
            'webchannel': [None, None, None]
        }
        df = pd.DataFrame(raw_data)

        result = prepare_show_data(df)

        # All ratings should be None/NaN
        assert pd.isna(result.iloc[0]['rating_avg'])
        assert pd.isna(result.iloc[1]['rating_avg'])
        assert pd.isna(result.iloc[2]['rating_avg'])

    def test_prepare_show_data_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=[
            'id', 'name', 'genres', 'summary', 'type', 'language',
            'status', 'rating', 'network', 'webchannel'
        ])

        result = prepare_show_data(df)

        assert len(result) == 0
        assert 'summary_clean' in result.columns
        assert 'platform' in result.columns


class TestSavePreparedData:
    """Tests for save_prepared_data function."""

    def test_save_prepared_data_success(self, temp_data_dir, sample_shows_df):
        """Test successful saving of prepared data."""
        output_dir = temp_data_dir / 'output'

        save_prepared_data(sample_shows_df, output_dir)

        # Verify directory was created
        assert output_dir.exists()
        assert output_dir.is_dir()

        # Verify CSV file was created
        csv_path = output_dir / 'shows_metadata.csv'
        assert csv_path.exists()

        # Verify data integrity
        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == len(sample_shows_df)
        assert list(loaded_df.columns) == list(sample_shows_df.columns)

    def test_save_prepared_data_creates_nested_directories(self, temp_data_dir, sample_shows_df):
        """Test creation of nested directories."""
        output_dir = temp_data_dir / 'level1' / 'level2' / 'level3'

        save_prepared_data(sample_shows_df, output_dir)

        assert output_dir.exists()
        assert (output_dir / 'shows_metadata.csv').exists()

    def test_save_prepared_data_overwrites_existing(self, temp_data_dir, sample_shows_df):
        """Test overwriting existing file."""
        output_dir = temp_data_dir / 'output'
        csv_path = output_dir / 'shows_metadata.csv'

        # Create initial file
        save_prepared_data(sample_shows_df, output_dir)
        initial_size = csv_path.stat().st_size

        # Modify data and save again
        modified_df = sample_shows_df.copy()
        modified_df['test_column'] = 'test'
        save_prepared_data(modified_df, output_dir)

        # Verify file was overwritten
        loaded_df = pd.read_csv(csv_path)
        assert 'test_column' in loaded_df.columns

    def test_save_prepared_data_empty_dataframe(self, temp_data_dir):
        """Test saving empty DataFrame."""
        output_dir = temp_data_dir / 'output'
        empty_df = pd.DataFrame(columns=['id', 'name'])

        save_prepared_data(empty_df, output_dir)

        csv_path = output_dir / 'shows_metadata.csv'
        assert csv_path.exists()

        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == 0
        assert list(loaded_df.columns) == ['id', 'name']


class TestMain:
    """Tests for main function."""

    @patch('scripts.fetch_and_prepare_data.save_prepared_data')
    @patch('scripts.fetch_and_prepare_data.prepare_show_data')
    @patch('scripts.fetch_and_prepare_data.fetch_all_shows')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_success_with_defaults(
        self, mock_parse_args, mock_fetch, mock_prepare, mock_save, sample_shows_list, temp_data_dir
    ):
        """Test main function with default arguments."""
        # Setup mocks
        mock_args = Mock()
        mock_args.output_dir = 'data/processed'
        mock_args.show_service_url = None
        mock_args.batch_size = 1000
        mock_args.max_shows = None
        mock_parse_args.return_value = mock_args

        mock_df_raw = pd.DataFrame(sample_shows_list + [
            {'id': i, 'name': f'Show{i}', 'genres': [], 'summary': None,
             'type': None, 'language': None, 'status': None,
             'rating': None, 'network': None, 'webchannel': None}
            for i in range(4, 10)
        ])
        mock_fetch.return_value = mock_df_raw

        mock_df_prepared = pd.DataFrame(sample_shows_list)
        mock_prepare.return_value = mock_df_prepared

        # Run main
        main()

        # Verify function calls
        mock_fetch.assert_called_once_with(
            show_service_url=None,
            batch_size=1000,
            max_shows=None
        )
        mock_prepare.assert_called_once()
        mock_save.assert_called_once()

    @patch('scripts.fetch_and_prepare_data.save_prepared_data')
    @patch('scripts.fetch_and_prepare_data.prepare_show_data')
    @patch('scripts.fetch_and_prepare_data.fetch_all_shows')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_success_with_custom_args(
        self, mock_parse_args, mock_fetch, mock_prepare, mock_save, sample_shows_list
    ):
        """Test main function with custom arguments."""
        mock_args = Mock()
        mock_args.output_dir = 'custom/output'
        mock_args.show_service_url = 'http://custom.com'
        mock_args.batch_size = 500
        mock_args.max_shows = 100
        mock_parse_args.return_value = mock_args

        mock_df_raw = pd.DataFrame(sample_shows_list)
        mock_fetch.return_value = mock_df_raw
        mock_df_prepared = pd.DataFrame(sample_shows_list)
        mock_prepare.return_value = mock_df_prepared

        main()

        mock_fetch.assert_called_once_with(
            show_service_url='http://custom.com',
            batch_size=500,
            max_shows=100
        )

    @patch('scripts.fetch_and_prepare_data.fetch_all_shows')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_main_handles_fetch_error(self, mock_exit, mock_parse_args, mock_fetch):
        """Test main function handles fetch errors."""
        mock_args = Mock()
        mock_args.output_dir = 'data/processed'
        mock_args.show_service_url = None
        mock_args.batch_size = 1000
        mock_args.max_shows = None
        mock_parse_args.return_value = mock_args

        mock_fetch.side_effect = Exception("API connection failed")

        main()

        # Verify sys.exit was called with error code
        mock_exit.assert_called_once_with(1)

    @patch('scripts.fetch_and_prepare_data.prepare_show_data')
    @patch('scripts.fetch_and_prepare_data.fetch_all_shows')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_main_handles_prepare_error(
        self, mock_exit, mock_parse_args, mock_fetch, mock_prepare, sample_shows_list
    ):
        """Test main function handles preparation errors."""
        mock_args = Mock()
        mock_args.output_dir = 'data/processed'
        mock_args.show_service_url = None
        mock_args.batch_size = 1000
        mock_args.max_shows = None
        mock_parse_args.return_value = mock_args

        mock_fetch.return_value = pd.DataFrame(sample_shows_list)
        mock_prepare.side_effect = Exception("Data preparation failed")

        main()

        mock_exit.assert_called_once_with(1)

    @patch('scripts.fetch_and_prepare_data.save_prepared_data')
    @patch('scripts.fetch_and_prepare_data.prepare_show_data')
    @patch('scripts.fetch_and_prepare_data.fetch_all_shows')
    @patch('argparse.ArgumentParser.parse_args')
    @patch('sys.exit')
    def test_main_handles_save_error(
        self, mock_exit, mock_parse_args, mock_fetch, mock_prepare, mock_save, sample_shows_list
    ):
        """Test main function handles save errors."""
        mock_args = Mock()
        mock_args.output_dir = 'data/processed'
        mock_args.show_service_url = None
        mock_args.batch_size = 1000
        mock_args.max_shows = None
        mock_parse_args.return_value = mock_args

        mock_fetch.return_value = pd.DataFrame(sample_shows_list)
        mock_prepare.return_value = pd.DataFrame(sample_shows_list)
        mock_save.side_effect = Exception("Save failed")

        main()

        mock_exit.assert_called_once_with(1)

    @patch('scripts.fetch_and_prepare_data.save_prepared_data')
    @patch('scripts.fetch_and_prepare_data.prepare_show_data')
    @patch('scripts.fetch_and_prepare_data.fetch_all_shows')
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_with_zero_shows(self, mock_parse_args, mock_fetch, mock_prepare, mock_save):
        """Test main function with zero shows fetched."""
        mock_args = Mock()
        mock_args.output_dir = 'data/processed'
        mock_args.show_service_url = None
        mock_args.batch_size = 1000
        mock_args.max_shows = None
        mock_parse_args.return_value = mock_args

        mock_fetch.return_value = pd.DataFrame()
        mock_prepare.return_value = pd.DataFrame()

        main()

        # Should still complete successfully
        mock_save.assert_called_once()
