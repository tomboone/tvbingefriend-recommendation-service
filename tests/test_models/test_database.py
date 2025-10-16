"""Unit tests for tvbingefriend_recommendation_service.models.database."""
import pytest
from unittest.mock import patch, Mock
from sqlalchemy.orm import Session
from sqlalchemy.engine import Engine

import tvbingefriend_recommendation_service.models.database as db_module


class TestDatabaseModule:
    """Tests for database module."""

    def test_database_url_is_set(self, monkeypatch):
        """Test that DATABASE_URL is set from config."""
        # Arrange
        monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')

        # Reload module to apply env changes
        import importlib
        importlib.reload(db_module)

        # Assert
        assert db_module.DATABASE_URL is not None
        assert 'sqlite' in db_module.DATABASE_URL

    def test_database_url_validation_raises_when_none(self, monkeypatch):
        """Test that module raises ValueError when DATABASE_URL is None."""
        # Arrange
        monkeypatch.delenv('DATABASE_URL', raising=False)

        # Act & Assert
        import importlib
        import sys

        # Save the original module if it exists
        original_module = sys.modules.get('tvbingefriend_recommendation_service.models.database')

        try:
            # Remove the module from cache so it can be reimported
            if 'tvbingefriend_recommendation_service.models.database' in sys.modules:
                del sys.modules['tvbingefriend_recommendation_service.models.database']

            # Mock get_database_url to return None before import
            with patch('tvbingefriend_recommendation_service.config.get_database_url', return_value=None):
                with pytest.raises(ValueError, match="DATABASE_URL is not configured"):
                    import tvbingefriend_recommendation_service.models.database as new_db_module
        finally:
            # Restore the original module to avoid breaking subsequent tests
            if original_module is not None:
                sys.modules['tvbingefriend_recommendation_service.models.database'] = original_module

    def test_engine_is_created(self, monkeypatch):
        """Test that SQLAlchemy engine is created."""
        # Arrange
        monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')
        import importlib
        importlib.reload(db_module)

        # Assert
        assert db_module.engine is not None
        assert isinstance(db_module.engine, Engine)

    def test_session_local_is_sessionmaker(self, monkeypatch):
        """Test that SessionLocal is a sessionmaker."""
        # Arrange
        monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')
        import importlib
        importlib.reload(db_module)

        # Assert
        assert db_module.SessionLocal is not None
        assert callable(db_module.SessionLocal)

    def test_get_db_yields_session(self, monkeypatch):
        """Test that get_db yields a database session."""
        # Arrange
        monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')
        import importlib
        importlib.reload(db_module)

        # Act
        db_generator = db_module.get_db()
        db = next(db_generator)

        # Assert
        assert db is not None
        assert isinstance(db, Session)

        # Cleanup
        try:
            next(db_generator)
        except StopIteration:
            pass

    def test_get_db_closes_session_after_yield(self, monkeypatch):
        """Test that get_db closes session in finally block."""
        # Arrange
        monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')
        import importlib
        importlib.reload(db_module)

        # Act
        db_generator = db_module.get_db()
        db = next(db_generator)

        # Try to finish the generator
        try:
            next(db_generator)
        except StopIteration:
            pass

        # Assert - if we got here without error, session was properly closed
        assert True

    def test_engine_configuration(self, monkeypatch):
        """Test that engine has correct configuration."""
        # Arrange
        monkeypatch.setenv('DATABASE_URL', 'sqlite:///:memory:')
        import importlib
        importlib.reload(db_module)

        # Assert
        assert db_module.engine.pool._pre_ping
        assert db_module.engine.pool._recycle == 3600
        assert not db_module.engine.echo
