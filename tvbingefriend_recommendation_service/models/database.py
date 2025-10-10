"""tv_recommender_service/models/database.py"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tvbingefriend_recommendation_service.config import get_database_url

# Get database URL
DATABASE_URL = get_database_url()

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
