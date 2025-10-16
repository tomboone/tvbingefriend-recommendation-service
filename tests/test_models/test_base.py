"""Unit tests for tvbingefriend_recommendation_service.models.base."""
import pytest
from sqlalchemy.ext.declarative import DeclarativeMeta

from tvbingefriend_recommendation_service.models.base import Base


class TestBase:
    """Tests for Base declarative base."""

    def test_base_is_declarative_base(self):
        """Test that Base is a declarative base."""
        # Assert
        assert hasattr(Base, 'metadata')
        assert hasattr(Base, 'registry')
        assert isinstance(Base, DeclarativeMeta)

    def test_base_metadata_exists(self):
        """Test that Base has metadata attribute."""
        # Assert
        assert Base.metadata is not None
        assert hasattr(Base.metadata, 'tables')

    def test_base_registry_exists(self):
        """Test that Base has registry attribute."""
        # Assert
        assert Base.registry is not None
        assert hasattr(Base.registry, 'mappers')

    def test_base_can_be_inherited(self):
        """Test that Base can be used as a parent class."""
        # Arrange & Act
        from sqlalchemy import Column, Integer

        class TestModel(Base):
            __tablename__ = 'test_model'
            id = Column(Integer, primary_key=True)

        # Assert
        assert issubclass(TestModel, Base)
        assert TestModel.__tablename__ == 'test_model'
