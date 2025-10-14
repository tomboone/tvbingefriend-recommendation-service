"""Text processing utilities for TV show descriptions."""
import re
import pandas as pd
from typing import List, Optional


def clean_html(text: str | None) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Raw text possibly containing HTML (can be None)

    Returns:
        Cleaned text without HTML tags
    """
    if pd.isna(text):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', str(text))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_texts(texts: List[Optional[str]]) -> List[str]:
    """
    Clean a list of texts by removing HTML tags.

    Args:
        texts: List of raw texts

    Returns:
        List of cleaned texts
    """
    return [clean_html(text) for text in texts]


def combine_text_features(
    summary: str,
    season_summaries: Optional[List[str]] = None
) -> str:
    """
    Combine show summary with season summaries into a single text.

    Args:
        summary: Main show summary
        season_summaries: Optional list of season summaries

    Returns:
        Combined text
    """
    texts = [clean_html(summary)]

    if season_summaries:
        texts.extend([clean_html(s) for s in season_summaries if s])

    return " ".join(texts)
