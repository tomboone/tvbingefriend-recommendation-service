"""Unit tests for tvbingefriend_recommendation_service.ml.text_processor."""
import pytest
import pandas as pd
from tvbingefriend_recommendation_service.ml.text_processor import (
    clean_html,
    clean_texts,
    combine_text_features
)


class TestCleanHtml:
    """Tests for clean_html function."""

    def test_clean_html_with_tags(self):
        """Test cleaning HTML tags from text."""
        # Arrange
        text = "<p>This is <b>bold</b> text with <a href='link'>links</a>.</p>"
        expected = "This is bold text with links."

        # Act
        result = clean_html(text)

        # Assert
        assert result == expected

    def test_clean_html_with_none(self):
        """Test clean_html with None input."""
        # Arrange
        text = None

        # Act
        result = clean_html(text)

        # Assert
        assert result == ""

    def test_clean_html_with_nan(self):
        """Test clean_html with NaN input."""
        # Arrange
        text = pd.NA

        # Act
        result = clean_html(text)

        # Assert
        assert result == ""

    def test_clean_html_with_extra_whitespace(self):
        """Test cleaning extra whitespace."""
        # Arrange
        text = "<p>Text   with    extra     spaces</p>"
        expected = "Text with extra spaces"

        # Act
        result = clean_html(text)

        # Assert
        assert result == expected

    def test_clean_html_no_tags(self):
        """Test with text without HTML tags."""
        # Arrange
        text = "Plain text without tags"

        # Act
        result = clean_html(text)

        # Assert
        assert result == text

    def test_clean_html_empty_string(self):
        """Test with empty string."""
        # Arrange
        text = ""

        # Act
        result = clean_html(text)

        # Assert
        assert result == ""

    def test_clean_html_complex_nested_tags(self):
        """Test with complex nested HTML tags."""
        # Arrange
        text = "<div><p>Paragraph <span>with <strong>nested</strong> tags</span></p></div>"
        expected = "Paragraph with nested tags"

        # Act
        result = clean_html(text)

        # Assert
        assert result == expected


class TestCleanTexts:
    """Tests for clean_texts function."""

    def test_clean_texts_multiple_strings(self):
        """Test cleaning a list of HTML texts."""
        # Arrange
        texts = [
            "<p>First text</p>",
            "<b>Second text</b>",
            "Third text"
        ]
        expected = ["First text", "Second text", "Third text"]

        # Act
        result = clean_texts(texts)

        # Assert
        assert result == expected

    def test_clean_texts_with_none_values(self):
        """Test cleaning texts with None values."""
        # Arrange
        texts = ["<p>First</p>", None, "<b>Third</b>"]
        expected = ["First", "", "Third"]

        # Act
        result = clean_texts(texts)

        # Assert
        assert result == expected

    def test_clean_texts_empty_list(self):
        """Test with empty list."""
        # Arrange
        texts = []

        # Act
        result = clean_texts(texts)

        # Assert
        assert result == []


class TestCombineTextFeatures:
    """Tests for combine_text_features function."""

    def test_combine_text_features_with_summaries(self):
        """Test combining summary with season summaries."""
        # Arrange
        summary = "<p>Main summary</p>"
        season_summaries = ["<p>Season 1</p>", "<p>Season 2</p>"]
        expected = "Main summary Season 1 Season 2"

        # Act
        result = combine_text_features(summary, season_summaries)

        # Assert
        assert result == expected

    def test_combine_text_features_no_season_summaries(self):
        """Test with only main summary."""
        # Arrange
        summary = "<p>Main summary</p>"
        expected = "Main summary"

        # Act
        result = combine_text_features(summary)

        # Assert
        assert result == expected

    def test_combine_text_features_with_none_in_seasons(self):
        """Test filtering out None values in season summaries."""
        # Arrange
        summary = "Main summary"
        season_summaries = ["Season 1", None, "Season 3"]
        expected = "Main summary Season 1 Season 3"

        # Act
        result = combine_text_features(summary, season_summaries)

        # Assert
        assert result == expected

    def test_combine_text_features_empty_season_list(self):
        """Test with empty season summaries list."""
        # Arrange
        summary = "Main summary"
        season_summaries = []
        expected = "Main summary"

        # Act
        result = combine_text_features(summary, season_summaries)

        # Assert
        assert result == expected

    def test_combine_text_features_html_in_both(self):
        """Test with HTML in both summary and season summaries."""
        # Arrange
        summary = "<div><p>Main</p></div>"
        season_summaries = ["<b>Season 1</b>", "<i>Season 2</i>"]
        expected = "Main Season 1 Season 2"

        # Act
        result = combine_text_features(summary, season_summaries)

        # Assert
        assert result == expected
