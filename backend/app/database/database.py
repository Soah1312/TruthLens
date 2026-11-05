from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.core.config import settings

# SQLite connection configuration:
# check_same_thread=False is required for SQLite when accessed by FastAPI worker threads
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# Create SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=(settings.ENVIRONMENT == "development"),
)

# Create sessionmaker factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Declarative base class for future Phase ORM models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency provider yielding a database session per request lifecycle.
    Ensures sessions are cleanly closed even if exceptions occur.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
