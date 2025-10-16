"""SQLAlchemy models"""

from tvbingefriend_recommendation_service.models.base import Base
from tvbingefriend_recommendation_service.models.show_metdata import ShowMetadata
from tvbingefriend_recommendation_service.models.show_similarity import ShowSimilarity

__all__ = [
    "Base",
    "ShowMetadata",
    "ShowSimilarity",
]
