"""Stores pre-computed similarity scores between shows."""

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Float, Index, Integer

from tvbingefriend_recommendation_service.models.base import Base


class ShowSimilarity(Base):
    """Stores pre-computed similarity scores between shows.

    Each row represents one show's similarity to another show.
    """

    __tablename__ = "show_similarities"

    # Composite primary key
    show_id = Column(Integer, primary_key=True)
    similar_show_id = Column(Integer, primary_key=True)

    # Similarity scores
    similarity_score = Column(Float, nullable=False)  # Combined/hybrid score
    genre_score = Column(Float, nullable=True)
    text_score = Column(Float, nullable=True)
    metadata_score = Column(Float, nullable=True)

    # Metadata
    computed_at = Column(DateTime, default=datetime.now(UTC), nullable=False)

    # Index for fast lookups
    __table_args__ = (
        Index("idx_show_id", "show_id"),
        Index("idx_similarity_score", "show_id", "similarity_score"),
    )

    def __repr__(self):
        return (
            f"<ShowSimilarity(show_id={self.show_id}, similar_show_id={self.similar_show_id}, "
            f"score={self.similarity_score:.3f})>"
        )
