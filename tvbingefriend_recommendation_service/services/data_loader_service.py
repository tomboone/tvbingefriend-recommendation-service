"""Service to load show data from existing show/season/episode/microservices"""
from typing import List, Dict, Optional
import time
import logging
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tvbingefriend_recommendation_service.config import get_service_url

logger = logging.getLogger(__name__)


class ShowDataLoader:
    """Service to load show data from existing show/season/episode microservices."""

    def __init__(
            self,
            show_service_url: Optional[str] = None,
            season_service_url: Optional[str] = None,
            episode_service_url: Optional[str] = None
    ):
        # Default to localhost for development
        self.show_service_url = show_service_url or get_service_url('show', 7071)
        self.season_service_url = season_service_url or get_service_url('season', 7072)
        self.episode_service_url = episode_service_url or get_service_url('episode', 7073)

        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # noinspection HttpUrlsUsage
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # ===== SHOW ENDPOINTS =====

    def get_show(self, show_id: int) -> Dict:
        """Fetch single show by ID"""
        url = f"{self.show_service_url}/shows/{show_id}"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_all_shows_bulk(self, offset: int = 0, limit: int = 100) -> Dict:
        """
        Fetch shows using the bulk endpoint with pagination.

        Returns:
            {
                "shows": [...],
                "total": 12345,
                "offset": 0,
                "limit": 100
            }
        """
        url = f"{self.show_service_url}/get_shows_bulk"
        params = {'offset': offset, 'limit': limit}
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_all_shows(self, batch_size: int = 100, max_shows: Optional[int] = None) -> List[Dict]:
        """
        Fetch all shows using pagination.

        Args:
            batch_size: Number of shows per request
            max_shows: Optional limit on total shows to fetch (for testing)

        Returns:
            List of show dictionaries
        """
        all_shows: List[Dict] = []
        offset = 0

        logger.info(f"Fetching all shows (batch size: {batch_size})...")

        while True:
            if max_shows and len(all_shows) >= max_shows:
                logger.info(f"Reached max_shows limit: {max_shows}")
                break

            result = self.get_all_shows_bulk(offset=offset, limit=batch_size)
            shows = result.get('shows', [])

            if not shows:
                break

            all_shows.extend(shows)
            logger.info(f"  Loaded {len(all_shows)} shows...")

            # Check if we've gotten all results (fewer shows than requested means we're at the end)
            if len(shows) < batch_size:
                break

            offset += batch_size
            time.sleep(0.1)  # Rate limiting

        logger.info(f"✓ Loaded {len(all_shows)} total shows")
        return all_shows

    # ===== SEASON ENDPOINTS =====

    def get_season(self, season_id: int) -> Dict:
        """Fetch single season by ID"""
        url = f"{self.season_service_url}/seasons/{season_id}"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_seasons_by_show(self, show_id: int) -> List[Dict]:
        """Fetch all seasons for a show"""
        url = f"{self.season_service_url}/shows/{show_id}/seasons"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_season_by_show_and_number(self, show_id: int, season_number: int) -> Dict:
        """Fetch specific season by show ID and season number"""
        url = f"{self.season_service_url}/shows/{show_id}/seasons/{season_number}"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    # ===== EPISODE ENDPOINTS =====

    def get_episode(self, episode_id: int) -> Dict:
        """Fetch single episode by ID"""
        url = f"{self.episode_service_url}/episodes/{episode_id}"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_episodes_by_season(self, show_id: int, season_number: int) -> List[Dict]:
        """Fetch all episodes for a specific season"""
        url = f"{self.episode_service_url}/shows/{show_id}/seasons/{season_number}/episodes"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return response.json()

    # ===== ENRICHMENT OPERATIONS =====

    def get_show_with_seasons(self, show_id: int) -> Dict:
        """Fetch show with all its seasons"""
        show = self.get_show(show_id)
        show['seasons'] = self.get_seasons_by_show(show_id)
        return show

    def enrich_shows_with_seasons(
            self,
            batch_size: int = 100,
            max_shows: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch all shows and enrich each with season summaries.
        Optimized for content-based recommendation feature extraction.

        Returns list of enriched show dicts with structure:
        {
            'show_id': int,
            'name': str,
            'summary': str,
            'genres': list,
            'network': str,
            'rating': float,
            'type': str,
            'language': str,
            'status': str,
            'season_summaries': list[str],
            'season_count': int,
            ...
        }
        """
        # Get all shows
        all_shows = self.get_all_shows(batch_size=batch_size, max_shows=max_shows)
        enriched_shows = []

        logger.info(f"Enriching {len(all_shows)} shows with season data...")

        for i, show in enumerate(all_shows):
            show_id = show['id']

            try:
                # Fetch seasons for this show
                seasons = self.get_seasons_by_show(show_id)

                # Extract season summaries (if your season model has summary field)
                season_summaries = [
                    s.get('summary', '')
                    for s in seasons
                    if s.get('summary')
                ]

                # Create enriched show record
                enriched = {
                    'show_id': show_id,
                    'name': show['name'],
                    'summary': show.get('summary', ''),
                    'genres': show.get('genres', []),
                    'network': show.get('network', {}).get('name') if show.get('network') else None,
                    'webchannel': show.get('webchannel', {}).get('name') if show.get('webchannel') else None,
                    'rating': show.get('rating', {}).get('average'),
                    'type': show.get('type'),
                    'language': show.get('language'),
                    'status': show.get('status'),
                    'season_summaries': season_summaries,
                    'season_count': len(seasons),
                    'premiered': show.get('premiered'),
                    'ended': show.get('ended'),
                }

                enriched_shows.append(enriched)

                if (i + 1) % 50 == 0:
                    logger.info(f"  Enriched {i + 1}/{len(all_shows)} shows...")

                time.sleep(0.05)  # Rate limiting

            except requests.HTTPError as e:
                logger.warning(f"Failed to enrich show {show_id} ({show['name']}): {e}")
                # Still include the show with basic data
                enriched_shows.append({
                    'show_id': show_id,
                    'name': show['name'],
                    'summary': show.get('summary', ''),
                    'genres': show.get('genres', []),
                    'network': show.get('network', {}).get('name') if show.get('network') else None,
                    'rating': show.get('rating', {}).get('average'),
                    'type': show.get('type'),
                    'language': show.get('language'),
                    'status': show.get('status'),
                    'season_summaries': [],
                    'season_count': 0,
                })

        logger.info(f"✓ Enriched {len(enriched_shows)} shows with season data")
        return enriched_shows
