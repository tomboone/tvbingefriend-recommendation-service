"""Get recommendations for a show."""
import azure.functions as func
import logging
import json

from tvbingefriend_recommendation_service.services import ContentBasedRecommendationService

# Initialize blueprint
bp = func.Blueprint()

# Initialize service (singleton pattern)
recommendation_service = ContentBasedRecommendationService()

logger = logging.getLogger(__name__)


@bp.route(route="shows/{show_id}/recommendations", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def get_show_recommendations(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get recommendations for a specific show.

    Query Parameters:
        - n: Number of recommendations (default: 10, max: 50)
        - min_similarity: Minimum similarity threshold (default: 0.0)
    """
    try:
        # Get show_id from route
        show_id = req.route_params.get('show_id')

        if not show_id:
            return func.HttpResponse(
                json.dumps({"error": "show_id is required"}),
                status_code=400,
                mimetype="application/json"
            )

        try:
            show_id = int(show_id)
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "show_id must be an integer"}),
                status_code=400,
                mimetype="application/json"
            )

        # Get query parameters
        n = int(req.params.get('n', 10))
        min_similarity = float(req.params.get('min_similarity', 0.0))

        # Validate parameters
        if n < 1 or n > 50:
            return func.HttpResponse(
                json.dumps({"error": "n must be between 1 and 50"}),
                status_code=400,
                mimetype="application/json"
            )

        if min_similarity < 0 or min_similarity > 1:
            return func.HttpResponse(
                json.dumps({"error": "min_similarity must be between 0 and 1"}),
                status_code=400,
                mimetype="application/json"
            )

        # Get recommendations from database
        recommendations = recommendation_service.get_recommendations_from_db(
            show_id=show_id,
            n=n,
            min_similarity=min_similarity
        )

        if not recommendations:
            return func.HttpResponse(
                json.dumps({
                    "show_id": show_id,
                    "recommendations": [],
                    "message": "No recommendations found for this show"
                }),
                status_code=404,
                mimetype="application/json"
            )

        # Format response
        response = {
            "show_id": show_id,
            "count": len(recommendations),
            "recommendations": recommendations
        }

        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": "Internal server error"}),
            status_code=500,
            mimetype="application/json"
        )


# noinspection PyUnusedLocal
@bp.route(route="recommendations/stats", methods=["GET"])
def get_recommendation_stats(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get statistics about the recommendation system.
    """
    try:
        stats = recommendation_service.get_stats()

        return func.HttpResponse(
            json.dumps(stats, default=str),  # default=str handles datetime
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        return func.HttpResponse(
            json.dumps({"error": "Internal server error"}),
            status_code=500,
            mimetype="application/json"
        )


# noinspection PyUnusedLocal
@bp.route(route="recommendations/health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint."""
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "service": "tv-recommender-service",
            "version": "1.0.0"
        }),
        status_code=200,
        mimetype="application/json"
    )
