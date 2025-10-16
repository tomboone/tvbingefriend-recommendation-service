"""Azure function blueprints"""

from tvbingefriend_recommendation_service.blueprints.recommendations_bp import (
    bp as recommendations_bp,
)

__all__ = [
    "recommendations_bp",
]
