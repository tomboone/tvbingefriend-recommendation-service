"""Integration tests for recommendations blueprint Azure Functions."""
import pytest
import json
from unittest.mock import Mock, patch
import azure.functions as func

from tvbingefriend_recommendation_service.blueprints.recommendations_bp import (
    get_show_recommendations,
    get_recommendation_stats,
    health_check,
    bp,
    recommendation_service
)


class TestGetShowRecommendations:
    """Tests for get_show_recommendations function."""

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_get_show_recommendations_returns_recommendations(self, mock_service):
        """Test getting recommendations for a valid show."""
        # Arrange
        mock_service.get_recommendations_from_db.return_value = [
            {
                'show_id': 2,
                'name': 'Similar Show',
                'genres': ['Drama'],
                'similarity_score': 0.85
            }
        ]

        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '1'}
        mock_req.params = {}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 200
        assert response.mimetype == "application/json"

        body = json.loads(response.get_body())
        assert body['show_id'] == 1
        assert body['count'] == 1
        assert len(body['recommendations']) == 1
        assert body['recommendations'][0]['show_id'] == 2

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_get_show_recommendations_with_custom_n_parameter(self, mock_service):
        """Test getting recommendations with custom n parameter."""
        # Arrange
        mock_service.get_recommendations_from_db.return_value = [
            {'show_id': i, 'name': f'Show {i}', 'genres': [], 'similarity_score': 0.8}
            for i in range(2, 7)
        ]

        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '1'}
        mock_req.params = {'n': '5'}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 200
        body = json.loads(response.get_body())
        assert body['count'] == 5
        mock_service.get_recommendations_from_db.assert_called_once_with(show_id=1, n=5, min_similarity=0.0)

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_get_show_recommendations_with_min_similarity_parameter(self, mock_service):
        """Test getting recommendations with min_similarity parameter."""
        # Arrange
        mock_service.get_recommendations_from_db.return_value = [
            {'show_id': 2, 'name': 'Show 2', 'genres': [], 'similarity_score': 0.9}
        ]

        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '1'}
        mock_req.params = {'min_similarity': '0.5'}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 200
        mock_service.get_recommendations_from_db.assert_called_once_with(show_id=1, n=10, min_similarity=0.5)

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_get_show_recommendations_returns_404_when_no_recommendations(self, mock_service):
        """Test 404 response when no recommendations found."""
        # Arrange
        mock_service.get_recommendations_from_db.return_value = []

        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '999'}
        mock_req.params = {}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 404
        body = json.loads(response.get_body())
        assert body['show_id'] == 999
        assert body['recommendations'] == []
        assert 'message' in body

    def test_get_show_recommendations_returns_400_when_show_id_missing(self):
        """Test 400 response when show_id is missing."""
        # Arrange
        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {}
        mock_req.params = {}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 400
        body = json.loads(response.get_body())
        assert 'error' in body
        assert 'show_id is required' in body['error']

    def test_get_show_recommendations_returns_400_when_show_id_invalid(self):
        """Test 400 response when show_id is not an integer."""
        # Arrange
        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': 'invalid'}
        mock_req.params = {}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 400
        body = json.loads(response.get_body())
        assert 'error' in body
        assert 'must be an integer' in body['error']

    def test_get_show_recommendations_returns_400_when_n_out_of_range_too_low(self):
        """Test 400 response when n is less than 1."""
        # Arrange
        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '1'}
        mock_req.params = {'n': '0'}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 400
        body = json.loads(response.get_body())
        assert 'error' in body
        assert 'n must be between 1 and 50' in body['error']

    def test_get_show_recommendations_returns_400_when_n_out_of_range_too_high(self):
        """Test 400 response when n is greater than 50."""
        # Arrange
        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '1'}
        mock_req.params = {'n': '51'}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 400
        body = json.loads(response.get_body())
        assert 'error' in body
        assert 'n must be between 1 and 50' in body['error']

    def test_get_show_recommendations_returns_400_when_min_similarity_too_low(self):
        """Test 400 response when min_similarity is less than 0."""
        # Arrange
        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '1'}
        mock_req.params = {'min_similarity': '-0.1'}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 400
        body = json.loads(response.get_body())
        assert 'error' in body
        assert 'min_similarity must be between 0 and 1' in body['error']

    def test_get_show_recommendations_returns_400_when_min_similarity_too_high(self):
        """Test 400 response when min_similarity is greater than 1."""
        # Arrange
        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '1'}
        mock_req.params = {'min_similarity': '1.5'}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 400
        body = json.loads(response.get_body())
        assert 'error' in body
        assert 'min_similarity must be between 0 and 1' in body['error']

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_get_show_recommendations_returns_500_on_service_error(self, mock_service):
        """Test 500 response when service raises an error."""
        # Arrange
        mock_service.get_recommendations_from_db.side_effect = Exception("Database error")

        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '1'}
        mock_req.params = {}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 500
        body = json.loads(response.get_body())
        assert 'error' in body
        assert 'Internal server error' in body['error']

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_get_show_recommendations_handles_value_error_in_parameters(self, mock_service):
        """Test handling of ValueError when parsing parameters."""
        # Arrange
        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '1'}
        mock_req.params = {'n': 'not-a-number'}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 500
        body = json.loads(response.get_body())
        assert 'error' in body


class TestGetRecommendationStats:
    """Tests for get_recommendation_stats function."""

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_get_recommendation_stats_returns_statistics(self, mock_service):
        """Test getting recommendation statistics."""
        # Arrange
        mock_service.get_stats.return_value = {
            'similarity_stats': {
                'total_records': 100,
                'unique_shows': 50
            },
            'cached_shows': 50,
            'weights': {
                'genre': 0.4,
                'text': 0.5,
                'metadata': 0.1
            }
        }

        mock_req = Mock(spec=func.HttpRequest)

        # Act
        response = get_recommendation_stats(mock_req)

        # Assert
        assert response.status_code == 200
        assert response.mimetype == "application/json"

        body = json.loads(response.get_body())
        assert 'similarity_stats' in body
        assert 'cached_shows' in body
        assert 'weights' in body
        assert body['cached_shows'] == 50

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_get_recommendation_stats_handles_datetime_serialization(self, mock_service):
        """Test that datetime objects are serialized correctly."""
        # Arrange
        from datetime import datetime
        mock_service.get_stats.return_value = {
            'last_updated': datetime(2024, 1, 1, 12, 0, 0),
            'cached_shows': 50
        }

        mock_req = Mock(spec=func.HttpRequest)

        # Act
        response = get_recommendation_stats(mock_req)

        # Assert
        assert response.status_code == 200
        # Should not raise an error due to datetime serialization
        body = json.loads(response.get_body())
        assert 'last_updated' in body

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_get_recommendation_stats_returns_500_on_service_error(self, mock_service):
        """Test 500 response when service raises an error."""
        # Arrange
        mock_service.get_stats.side_effect = Exception("Database error")

        mock_req = Mock(spec=func.HttpRequest)

        # Act
        response = get_recommendation_stats(mock_req)

        # Assert
        assert response.status_code == 500
        body = json.loads(response.get_body())
        assert 'error' in body
        assert 'Internal server error' in body['error']


class TestHealthCheck:
    """Tests for health_check function."""

    def test_health_check_returns_healthy_status(self):
        """Test health check endpoint returns healthy status."""
        # Arrange
        mock_req = Mock(spec=func.HttpRequest)

        # Act
        response = health_check(mock_req)

        # Assert
        assert response.status_code == 200
        assert response.mimetype == "application/json"

        body = json.loads(response.get_body())
        assert body['status'] == 'healthy'
        assert body['service'] == 'tv-recommender-service'
        assert 'version' in body

    def test_health_check_always_succeeds(self):
        """Test that health check always returns 200 even with invalid request."""
        # Arrange
        mock_req = None  # Invalid request

        # Act
        response = health_check(mock_req)

        # Assert
        assert response.status_code == 200


class TestBlueprintIntegration:
    """Integration tests for blueprint behavior."""

    def test_blueprint_is_registered(self):
        """Test that blueprint is properly initialized."""
        # Assert
        assert bp is not None

    def test_recommendation_service_exists(self):
        """Test that recommendation service is initialized."""
        # Assert
        assert recommendation_service is not None

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_end_to_end_recommendation_flow(self, mock_service):
        """Test complete flow of getting recommendations."""
        # Arrange
        mock_service.get_recommendations_from_db.return_value = [
            {
                'show_id': 2,
                'name': 'Breaking Bad',
                'genres': ['Drama', 'Crime'],
                'summary': 'A chemistry teacher...',
                'rating': 9.5,
                'similarity_score': 0.95,
                'genre_score': 0.9,
                'text_score': 0.95,
                'metadata_score': 1.0
            },
            {
                'show_id': 3,
                'name': 'Better Call Saul',
                'genres': ['Drama', 'Crime'],
                'summary': 'The story of Jimmy...',
                'rating': 8.8,
                'similarity_score': 0.85,
                'genre_score': 0.9,
                'text_score': 0.8,
                'metadata_score': 0.85
            }
        ]

        mock_req = Mock(spec=func.HttpRequest)
        mock_req.route_params = {'show_id': '1'}
        mock_req.params = {'n': '2', 'min_similarity': '0.5'}

        # Act
        response = get_show_recommendations(mock_req)

        # Assert
        assert response.status_code == 200
        body = json.loads(response.get_body())

        # Verify request was passed correctly
        mock_service.get_recommendations_from_db.assert_called_once_with(show_id=1, n=2, min_similarity=0.5)

        # Verify response structure
        assert body['show_id'] == 1
        assert body['count'] == 2
        assert len(body['recommendations']) == 2

        # Verify first recommendation
        rec1 = body['recommendations'][0]
        assert rec1['show_id'] == 2
        assert rec1['name'] == 'Breaking Bad'
        assert rec1['similarity_score'] == 0.95
        assert 'genre_score' in rec1
        assert 'text_score' in rec1

    @patch('tvbingefriend_recommendation_service.blueprints.recommendations_bp.recommendation_service')
    def test_end_to_end_stats_flow(self, mock_service):
        """Test complete flow of getting stats."""
        # Arrange
        mock_service.get_stats.return_value = {
            'similarity_stats': {
                'total_records': 1500,
                'unique_shows': 100,
                'avg_similarities_per_show': 15.0
            },
            'cached_shows': 100,
            'weights': {
                'genre': 0.4,
                'text': 0.5,
                'metadata': 0.1
            }
        }

        mock_req = Mock(spec=func.HttpRequest)

        # Act
        response = get_recommendation_stats(mock_req)

        # Assert
        assert response.status_code == 200
        body = json.loads(response.get_body())

        # Verify all expected fields
        assert body['similarity_stats']['total_records'] == 1500
        assert body['similarity_stats']['unique_shows'] == 100
        assert body['cached_shows'] == 100
        assert body['weights']['genre'] == 0.4
