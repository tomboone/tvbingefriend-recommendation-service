"""Unit tests for ShowDataLoader service."""

from unittest.mock import patch

import pytest
from requests.exceptions import HTTPError

from tvbingefriend_recommendation_service.services.data_loader_service import ShowDataLoader


class TestShowDataLoaderInit:
    """Tests for ShowDataLoader initialization."""

    def test_init_with_default_urls(self, mock_config, monkeypatch):
        """Test initialization with default URLs from config."""
        # Arrange
        monkeypatch.setenv("SHOW_SERVICE_URL", "http://show:7071/api")
        monkeypatch.setenv("SEASON_SERVICE_URL", "http://season:7072/api")
        monkeypatch.setenv("EPISODE_SERVICE_URL", "http://episode:7073/api")

        # Act
        loader = ShowDataLoader()

        # Assert
        assert "show" in loader.show_service_url
        assert "season" in loader.season_service_url
        assert "episode" in loader.episode_service_url
        assert loader.session is not None

    def test_init_with_custom_urls(self):
        """Test initialization with custom service URLs."""
        # Act
        loader = ShowDataLoader(
            show_service_url="http://custom-show:8080",
            season_service_url="http://custom-season:8081",
            episode_service_url="http://custom-episode:8082",
        )

        # Assert
        assert loader.show_service_url == "http://custom-show:8080"
        assert loader.season_service_url == "http://custom-season:8081"
        assert loader.episode_service_url == "http://custom-episode:8082"

    def test_init_configures_retry_strategy(self):
        """Test that session is configured with retry strategy."""
        # Act
        loader = ShowDataLoader()

        # Assert
        assert loader.session is not None
        # Verify adapters are mounted
        http_adapter = loader.session.get_adapter("http://test.com")
        https_adapter = loader.session.get_adapter("https://test.com")
        assert http_adapter is not None
        assert https_adapter is not None


class TestGetShow:
    """Tests for get_show method."""

    def test_get_show_returns_show_data(self, requests_mock):
        """Test fetching a single show by ID."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")
        show_data = {"id": 1, "name": "Breaking Bad"}
        requests_mock.get("http://test-show/shows/1", json=show_data)

        # Act
        result = loader.get_show(1)

        # Assert
        assert result == show_data
        assert result["id"] == 1
        assert result["name"] == "Breaking Bad"

    def test_get_show_raises_http_error_on_404(self, requests_mock):
        """Test that get_show raises HTTPError for 404 response."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")
        requests_mock.get("http://test-show/shows/999", status_code=404)

        # Act & Assert
        with pytest.raises(HTTPError):
            loader.get_show(999)

    def test_get_show_raises_http_error_on_500(self, requests_mock):
        """Test that get_show raises HTTPError for 500 response."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")
        requests_mock.get("http://test-show/shows/1", status_code=500)

        # Act & Assert
        with pytest.raises(HTTPError):
            loader.get_show(1)

    def test_get_show_uses_correct_timeout(self, requests_mock):
        """Test that get_show uses correct timeout."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")
        requests_mock.get("http://test-show/shows/1", json={"id": 1})

        # Act
        loader.get_show(1)

        # Assert
        assert requests_mock.last_request.timeout == 10


class TestGetAllShowsBulk:
    """Tests for get_all_shows_bulk method."""

    def test_get_all_shows_bulk_returns_paginated_data(self, requests_mock):
        """Test bulk endpoint returns paginated data."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")
        response_data = {
            "shows": [{"id": 1, "name": "Show 1"}, {"id": 2, "name": "Show 2"}],
            "total": 100,
            "offset": 0,
            "limit": 2,
        }
        requests_mock.get("http://test-show/get_shows_bulk", json=response_data)

        # Act
        result = loader.get_all_shows_bulk(offset=0, limit=2)

        # Assert
        assert result == response_data
        assert len(result["shows"]) == 2
        assert result["total"] == 100

    def test_get_all_shows_bulk_with_custom_offset_and_limit(self, requests_mock):
        """Test bulk endpoint with custom offset and limit."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")
        requests_mock.get("http://test-show/get_shows_bulk", json={"shows": []})

        # Act
        loader.get_all_shows_bulk(offset=50, limit=25)

        # Assert
        assert requests_mock.last_request.qs["offset"] == ["50"]
        assert requests_mock.last_request.qs["limit"] == ["25"]

    def test_get_all_shows_bulk_raises_http_error_on_failure(self, requests_mock):
        """Test bulk endpoint raises HTTPError on failure."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")
        requests_mock.get("http://test-show/get_shows_bulk", status_code=500)

        # Act & Assert
        with pytest.raises(HTTPError):
            loader.get_all_shows_bulk()


class TestGetAllShows:
    """Tests for get_all_shows method."""

    def test_get_all_shows_fetches_all_pages(self, requests_mock):
        """Test fetching all shows with pagination."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")

        # Mock first page
        requests_mock.get(
            "http://test-show/get_shows_bulk?offset=0&limit=2",
            json={"shows": [{"id": 1}, {"id": 2}], "total": 4, "offset": 0, "limit": 2},
        )
        # Mock second page
        requests_mock.get(
            "http://test-show/get_shows_bulk?offset=2&limit=2",
            json={"shows": [{"id": 3}, {"id": 4}], "total": 4, "offset": 2, "limit": 2},
        )
        # Mock third page (empty)
        requests_mock.get(
            "http://test-show/get_shows_bulk?offset=4&limit=2",
            json={"shows": [], "total": 4, "offset": 4, "limit": 2},
        )

        # Act
        result = loader.get_all_shows(batch_size=2)

        # Assert
        assert len(result) == 4
        assert result[0]["id"] == 1
        assert result[3]["id"] == 4

    def test_get_all_shows_stops_when_fewer_than_batch_size_returned(self, requests_mock):
        """Test that pagination stops when fewer results than batch size."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")

        # Mock first page (full)
        requests_mock.get(
            "http://test-show/get_shows_bulk?offset=0&limit=100",
            json={
                "shows": [{"id": i} for i in range(100)],
                "total": 150,
                "offset": 0,
                "limit": 100,
            },
        )
        # Mock second page (partial - indicates last page)
        requests_mock.get(
            "http://test-show/get_shows_bulk?offset=100&limit=100",
            json={
                "shows": [{"id": i} for i in range(100, 150)],
                "total": 150,
                "offset": 100,
                "limit": 100,
            },
        )

        # Act
        result = loader.get_all_shows(batch_size=100)

        # Assert
        assert len(result) == 150

    def test_get_all_shows_respects_max_shows_limit(self, requests_mock):
        """Test that max_shows parameter limits total shows fetched."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")

        # Mock multiple pages
        requests_mock.get(
            "http://test-show/get_shows_bulk?offset=0&limit=10",
            json={"shows": [{"id": i} for i in range(10)], "total": 100, "offset": 0, "limit": 10},
        )
        requests_mock.get(
            "http://test-show/get_shows_bulk?offset=10&limit=10",
            json={
                "shows": [{"id": i} for i in range(10, 20)],
                "total": 100,
                "offset": 10,
                "limit": 10,
            },
        )

        # Act
        result = loader.get_all_shows(batch_size=10, max_shows=15)

        # Assert
        # Note: The method fetches in batches, so it may fetch up to batch_size more than max_shows
        # This is expected behavior - it stops checking after >= max_shows
        assert len(result) >= 15
        assert len(result) <= 20  # Should not exceed next batch

    def test_get_all_shows_handles_empty_response(self, requests_mock):
        """Test handling of empty shows list."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")
        requests_mock.get(
            "http://test-show/get_shows_bulk?offset=0&limit=100",
            json={"shows": [], "total": 0, "offset": 0, "limit": 100},
        )

        # Act
        result = loader.get_all_shows()

        # Assert
        assert result == []

    @patch("time.sleep")
    def test_get_all_shows_includes_rate_limiting(self, mock_sleep, requests_mock):
        """Test that rate limiting is applied between requests."""
        # Arrange
        loader = ShowDataLoader(show_service_url="http://test-show")
        requests_mock.get(
            "http://test-show/get_shows_bulk?offset=0&limit=2",
            json={"shows": [{"id": 1}, {"id": 2}], "total": 3, "offset": 0, "limit": 2},
        )
        requests_mock.get(
            "http://test-show/get_shows_bulk?offset=2&limit=2",
            json={"shows": [{"id": 3}], "total": 3, "offset": 2, "limit": 2},
        )

        # Act
        loader.get_all_shows(batch_size=2)

        # Assert
        mock_sleep.assert_called_with(0.1)


class TestGetSeason:
    """Tests for get_season method."""

    def test_get_season_returns_season_data(self, requests_mock):
        """Test fetching a single season by ID."""
        # Arrange
        loader = ShowDataLoader(season_service_url="http://test-season")
        season_data = {"id": 1, "season_number": 1}
        requests_mock.get("http://test-season/seasons/1", json=season_data)

        # Act
        result = loader.get_season(1)

        # Assert
        assert result == season_data
        assert result["id"] == 1

    def test_get_season_raises_http_error_on_404(self, requests_mock):
        """Test that get_season raises HTTPError for 404."""
        # Arrange
        loader = ShowDataLoader(season_service_url="http://test-season")
        requests_mock.get("http://test-season/seasons/999", status_code=404)

        # Act & Assert
        with pytest.raises(HTTPError):
            loader.get_season(999)


class TestGetSeasonsByShow:
    """Tests for get_seasons_by_show method."""

    def test_get_seasons_by_show_returns_list_of_seasons(self, requests_mock):
        """Test fetching all seasons for a show."""
        # Arrange
        loader = ShowDataLoader(season_service_url="http://test-season")
        seasons_data = [{"id": 1, "season_number": 1}, {"id": 2, "season_number": 2}]
        requests_mock.get("http://test-season/shows/1/seasons", json=seasons_data)

        # Act
        result = loader.get_seasons_by_show(1)

        # Assert
        assert len(result) == 2
        assert result[0]["season_number"] == 1
        assert result[1]["season_number"] == 2

    def test_get_seasons_by_show_handles_empty_list(self, requests_mock):
        """Test handling of show with no seasons."""
        # Arrange
        loader = ShowDataLoader(season_service_url="http://test-season")
        requests_mock.get("http://test-season/shows/1/seasons", json=[])

        # Act
        result = loader.get_seasons_by_show(1)

        # Assert
        assert result == []


class TestGetSeasonByShowAndNumber:
    """Tests for get_season_by_show_and_number method."""

    def test_get_season_by_show_and_number_returns_season(self, requests_mock):
        """Test fetching specific season by show ID and season number."""
        # Arrange
        loader = ShowDataLoader(season_service_url="http://test-season")
        season_data = {"id": 1, "show_id": 1, "season_number": 2}
        requests_mock.get("http://test-season/shows/1/seasons/2", json=season_data)

        # Act
        result = loader.get_season_by_show_and_number(1, 2)

        # Assert
        assert result == season_data
        assert result["show_id"] == 1
        assert result["season_number"] == 2

    def test_get_season_by_show_and_number_raises_error_on_not_found(self, requests_mock):
        """Test error handling when season not found."""
        # Arrange
        loader = ShowDataLoader(season_service_url="http://test-season")
        requests_mock.get("http://test-season/shows/1/seasons/99", status_code=404)

        # Act & Assert
        with pytest.raises(HTTPError):
            loader.get_season_by_show_and_number(1, 99)


class TestGetEpisode:
    """Tests for get_episode method."""

    def test_get_episode_returns_episode_data(self, requests_mock):
        """Test fetching a single episode by ID."""
        # Arrange
        loader = ShowDataLoader(episode_service_url="http://test-episode")
        episode_data = {"id": 1, "name": "Pilot"}
        requests_mock.get("http://test-episode/episodes/1", json=episode_data)

        # Act
        result = loader.get_episode(1)

        # Assert
        assert result == episode_data
        assert result["name"] == "Pilot"

    def test_get_episode_raises_http_error_on_404(self, requests_mock):
        """Test that get_episode raises HTTPError for 404."""
        # Arrange
        loader = ShowDataLoader(episode_service_url="http://test-episode")
        requests_mock.get("http://test-episode/episodes/999", status_code=404)

        # Act & Assert
        with pytest.raises(HTTPError):
            loader.get_episode(999)


class TestGetEpisodesBySeason:
    """Tests for get_episodes_by_season method."""

    def test_get_episodes_by_season_returns_list_of_episodes(self, requests_mock):
        """Test fetching all episodes for a season."""
        # Arrange
        loader = ShowDataLoader(episode_service_url="http://test-episode")
        episodes_data = [{"id": 1, "name": "Episode 1"}, {"id": 2, "name": "Episode 2"}]
        requests_mock.get("http://test-episode/shows/1/seasons/1/episodes", json=episodes_data)

        # Act
        result = loader.get_episodes_by_season(1, 1)

        # Assert
        assert len(result) == 2
        assert result[0]["name"] == "Episode 1"

    def test_get_episodes_by_season_handles_empty_list(self, requests_mock):
        """Test handling of season with no episodes."""
        # Arrange
        loader = ShowDataLoader(episode_service_url="http://test-episode")
        requests_mock.get("http://test-episode/shows/1/seasons/1/episodes", json=[])

        # Act
        result = loader.get_episodes_by_season(1, 1)

        # Assert
        assert result == []


class TestGetShowWithSeasons:
    """Tests for get_show_with_seasons method."""

    def test_get_show_with_seasons_enriches_show_data(self, requests_mock):
        """Test fetching show and enriching with seasons."""
        # Arrange
        loader = ShowDataLoader(
            show_service_url="http://test-show", season_service_url="http://test-season"
        )

        show_data = {"id": 1, "name": "Breaking Bad"}
        seasons_data = [{"id": 1, "season_number": 1}]

        requests_mock.get("http://test-show/shows/1", json=show_data)
        requests_mock.get("http://test-season/shows/1/seasons", json=seasons_data)

        # Act
        result = loader.get_show_with_seasons(1)

        # Assert
        assert result["id"] == 1
        assert result["name"] == "Breaking Bad"
        assert "seasons" in result
        assert len(result["seasons"]) == 1
        assert result["seasons"][0]["season_number"] == 1


class TestEnrichShowsWithSeasons:
    """Tests for enrich_shows_with_seasons method."""

    def test_enrich_shows_with_seasons_creates_enriched_records(self, requests_mock):
        """Test enriching shows with season data."""
        # Arrange
        loader = ShowDataLoader(
            show_service_url="http://test-show", season_service_url="http://test-season"
        )

        # Mock shows
        shows_data = {
            "shows": [
                {
                    "id": 1,
                    "name": "Breaking Bad",
                    "summary": "A teacher cooks meth",
                    "genres": ["Drama"],
                    "network": {"name": "AMC"},
                    "webchannel": None,
                    "rating": {"average": 9.5},
                    "type": "Scripted",
                    "language": "English",
                    "status": "Ended",
                    "premiered": "2008-01-20",
                    "ended": "2013-09-29",
                }
            ],
            "total": 1,
            "offset": 0,
            "limit": 100,
        }

        # Mock seasons
        seasons_data = [{"id": 1, "season_number": 1, "summary": "Season 1 summary"}]

        requests_mock.get("http://test-show/get_shows_bulk", json=shows_data)
        requests_mock.get("http://test-season/shows/1/seasons", json=seasons_data)

        # Act
        result = loader.enrich_shows_with_seasons(batch_size=100, max_shows=1)

        # Assert
        assert len(result) == 1
        enriched = result[0]
        assert enriched["show_id"] == 1
        assert enriched["name"] == "Breaking Bad"
        assert enriched["season_summaries"] == ["Season 1 summary"]
        assert enriched["season_count"] == 1
        assert enriched["network"] == "AMC"
        assert enriched["rating"] == 9.5

    def test_enrich_shows_with_seasons_handles_missing_network(self, requests_mock):
        """Test enriching shows with missing network data."""
        # Arrange
        loader = ShowDataLoader(
            show_service_url="http://test-show", season_service_url="http://test-season"
        )

        shows_data = {
            "shows": [
                {
                    "id": 1,
                    "name": "Show",
                    "summary": "Summary",
                    "genres": [],
                    "network": None,
                    "webchannel": None,
                    "rating": {},  # Empty dict instead of None
                    "type": None,
                    "language": None,
                    "status": None,
                    "premiered": None,
                    "ended": None,
                }
            ],
            "total": 1,
            "offset": 0,
            "limit": 100,
        }

        requests_mock.get("http://test-show/get_shows_bulk", json=shows_data)
        requests_mock.get("http://test-season/shows/1/seasons", json=[])

        # Act
        result = loader.enrich_shows_with_seasons(max_shows=1)

        # Assert
        enriched = result[0]
        assert enriched["network"] is None
        assert enriched["webchannel"] is None
        assert enriched["rating"] is None

    def test_enrich_shows_with_seasons_uses_webchannel_when_no_network(self, requests_mock):
        """Test that webchannel is used when network is None."""
        # Arrange
        loader = ShowDataLoader(
            show_service_url="http://test-show", season_service_url="http://test-season"
        )

        shows_data = {
            "shows": [
                {
                    "id": 1,
                    "name": "Show",
                    "summary": "",
                    "genres": [],
                    "network": None,
                    "webchannel": {"name": "Netflix"},
                    "rating": {},
                    "type": None,
                    "language": None,
                    "status": None,
                    "premiered": None,
                    "ended": None,
                }
            ],
            "total": 1,
            "offset": 0,
            "limit": 100,
        }

        requests_mock.get("http://test-show/get_shows_bulk", json=shows_data)
        requests_mock.get("http://test-season/shows/1/seasons", json=[])

        # Act
        result = loader.enrich_shows_with_seasons(max_shows=1)

        # Assert
        assert result[0]["webchannel"] == "Netflix"

    def test_enrich_shows_with_seasons_handles_http_error_gracefully(self, requests_mock):
        """Test that enrichment continues even if season fetch fails."""
        # Arrange
        loader = ShowDataLoader(
            show_service_url="http://test-show", season_service_url="http://test-season"
        )

        shows_data = {
            "shows": [
                {
                    "id": 1,
                    "name": "Show",
                    "summary": "",
                    "genres": [],
                    "network": None,
                    "webchannel": None,
                    "rating": {},
                    "type": None,
                    "language": None,
                    "status": None,
                    "premiered": None,
                    "ended": None,
                }
            ],
            "total": 1,
            "offset": 0,
            "limit": 100,
        }

        requests_mock.get("http://test-show/get_shows_bulk", json=shows_data)
        requests_mock.get("http://test-season/shows/1/seasons", status_code=500)

        # Act
        result = loader.enrich_shows_with_seasons(max_shows=1)

        # Assert
        assert len(result) == 1
        assert result[0]["season_summaries"] == []
        assert result[0]["season_count"] == 0

    @patch("time.sleep")
    def test_enrich_shows_with_seasons_includes_rate_limiting(self, mock_sleep, requests_mock):
        """Test that rate limiting is applied during enrichment."""
        # Arrange
        loader = ShowDataLoader(
            show_service_url="http://test-show", season_service_url="http://test-season"
        )

        shows_data = {
            "shows": [
                {
                    "id": i,
                    "name": f"Show {i}",
                    "summary": "",
                    "genres": [],
                    "network": None,
                    "webchannel": None,
                    "rating": {},
                    "type": None,
                    "language": None,
                    "status": None,
                    "premiered": None,
                    "ended": None,
                }
                for i in range(1, 3)
            ],
            "total": 2,
            "offset": 0,
            "limit": 100,
        }

        requests_mock.get("http://test-show/get_shows_bulk", json=shows_data)
        requests_mock.get("http://test-season/shows/1/seasons", json=[])
        requests_mock.get("http://test-season/shows/2/seasons", json=[])

        # Act
        loader.enrich_shows_with_seasons(max_shows=2)

        # Assert
        # Should be called for each enrichment
        assert mock_sleep.call_count >= 2
