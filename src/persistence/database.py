"""Database engine and session management."""
import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)

_DEFAULT_URL = "postgresql://localhost:5432/trading"
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_engine(url: Optional[str] = None) -> Engine:
    global _engine
    if _engine is None:
        db_url = url or os.environ.get("DATABASE_URL", _DEFAULT_URL)
        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=os.environ.get("SQL_ECHO", "").lower() in ("1", "true"),
        )
        logger.info("Database engine created")
    return _engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        eng = engine or get_engine()
        _session_factory = sessionmaker(eng, autocommit=False, autoflush=False, expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Context manager for a single DB transaction. Commits on success, rolls back on exception."""
    factory = get_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
