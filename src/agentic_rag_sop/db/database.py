"""
Database connection and session lifecycle management using SQLAlchemy.
Supports SQLite by default and PostgreSQL via DATABASE_URL.
"""

from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from agentic_rag_sop.config.settings import get_settings

settings = get_settings()

db_url = settings.DATABASE_URL
connect_args = {}

if db_url.startswith("sqlite"):
    # If SQLite file path specified, make sure directory exists
    clean_path = db_url.replace("sqlite:///", "")
    if clean_path and clean_path != ":memory:":
        parent_dir = Path(clean_path).parent
        parent_dir.mkdir(parents=True, exist_ok=True)
    connect_args["check_same_thread"] = False

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db() -> None:
    """Create all database tables if they do not exist."""
    import agentic_rag_sop.db.models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
