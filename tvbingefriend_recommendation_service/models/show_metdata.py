"""Cached show metadata for quick access"""
from sqlalchemy import Column, Integer, Float, DateTime, String, Text
from sqlalchemy.dialects.mysql import JSON
from datetime import datetime, UTC

from tvbingefriend_recommendation_service.models.base import Base


class ShowMetadata(Base):
    """Cached show metadata for quick access.
    Mirrors data from your show service.
    """
    __tablename__ = 'show_metadata'

    show_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    genres = Column(JSON, nullable=True)
    summary = Column(Text, nullable=True)
    rating = Column(Float, nullable=True)
    type = Column(String(50), nullable=True)
    language = Column(String(50), nullable=True)
    network = Column(String(100), nullable=True)

    synced_at = Column(DateTime, default=datetime.now(UTC), nullable=False)

    def __repr__(self):
        return f"<ShowMetadata(show_id={self.show_id}, name='{self.name}')>"
