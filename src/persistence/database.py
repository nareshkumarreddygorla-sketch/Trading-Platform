"""Database engine and session management."""

import logging
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

_SQLITE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "trading.db")
_DEFAULT_URL = "postgresql://localhost:5432/trading"
_engine: Engine | None = None
_session_factory: sessionmaker | None = None
_engine_lock = threading.Lock()


def get_engine(url: str | None = None) -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            db_url = url or os.environ.get("DATABASE_URL")
            env = os.environ.get("ENV", os.environ.get("ENVIRONMENT", "development")).lower()
            if not db_url:
                if env == "production":
                    raise RuntimeError("DATABASE_URL is required in production. Set ENV=development for SQLite.")
                # Development/testing mode: use SQLite
                os.makedirs(os.path.dirname(_SQLITE_PATH), exist_ok=True)
                db_url = f"sqlite:///{_SQLITE_PATH}"
                logger.info("Using SQLite (development mode) at %s", _SQLITE_PATH)
            else:
                if db_url.startswith("sqlite"):
                    logger.info("Using SQLite (development mode)")
                else:
                    logger.info("Using PostgreSQL (production mode)")

            connect_args = {}
            kwargs: dict[str, Any] = {"echo": os.environ.get("SQL_ECHO", "").lower() in ("1", "true")}
            if db_url.startswith("sqlite"):
                connect_args["check_same_thread"] = False
            else:
                kwargs["pool_pre_ping"] = True
                kwargs["pool_size"] = int(os.environ.get("DB_POOL_SIZE", "10"))
                kwargs["max_overflow"] = int(os.environ.get("DB_MAX_OVERFLOW", "10"))
                kwargs["pool_recycle"] = 1800
                kwargs["pool_timeout"] = 30

            _engine = create_engine(db_url, connect_args=connect_args, **kwargs)
            logger.info("Database engine created (%s)", "sqlite" if db_url.startswith("sqlite") else "postgresql")
        return _engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        eng = engine or get_engine()
        _session_factory = sessionmaker(eng, autocommit=False, autoflush=False, expire_on_commit=False)
    return _session_factory


def check_database_health(engine: Engine | None = None, max_size_gb: float = 80.0) -> dict:
    """Check database health: connection, disk usage, and connection pool status.

    Returns dict with:
        healthy: bool - True if database is operational and disk is not full
        size_gb: float - Current database size in GB (PostgreSQL only)
        size_pct: float - Percentage of max_size_gb used
        pool_size: int - Current pool size
        pool_checked_out: int - Active connections
        error: str | None - Error message if unhealthy
    """
    result = {"healthy": True, "size_gb": 0.0, "size_pct": 0.0, "pool_size": 0, "pool_checked_out": 0, "error": None}
    try:
        eng = engine or get_engine()
        # Connection test
        from sqlalchemy import text

        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
            result["connectivity"] = True

        # Pool status and connection leak detection
        pool = eng.pool
        if hasattr(pool, "size"):
            result["pool_size"] = pool.size()
        if hasattr(pool, "checkedout"):
            checked_out = pool.checkedout()
            pool_size = pool.size()
            result["pool_checked_out"] = checked_out
            result["pool_size"] = pool_size
            if checked_out > pool_size * 0.8:
                logger.warning("Connection pool near exhaustion: %d/%d checked out", checked_out, pool_size)
                result["pool_warning"] = True

        # Database size check (PostgreSQL only)
        db_url = str(eng.url)
        if not db_url.startswith("sqlite"):
            try:
                from sqlalchemy import text

                with eng.connect() as conn:
                    row = conn.execute(text("SELECT pg_database_size(current_database())")).fetchone()
                    if row:
                        size_bytes = row[0]
                        result["size_gb"] = round(size_bytes / (1024**3), 2)
                        result["size_pct"] = round((result["size_gb"] / max_size_gb) * 100, 1)
                        if result["size_pct"] > 90:
                            result["healthy"] = False
                            result["error"] = (
                                f"Database disk usage critical: {result['size_gb']:.1f}GB ({result['size_pct']:.0f}% of {max_size_gb}GB limit)"
                            )
                            logger.critical(result["error"])
                        elif result["size_pct"] > 75:
                            logger.warning(
                                "Database disk usage high: %.1fGB (%.0f%% of limit)",
                                result["size_gb"],
                                result["size_pct"],
                            )
            except Exception as e:
                logger.debug("Database size check failed (non-critical): %s", e)
    except Exception as e:
        result["healthy"] = False
        result["error"] = f"Database connection failed: {e}"
        logger.error(result["error"])
    return result


@contextmanager
def session_scope(engine: Engine | None = None) -> Generator[Session, None, None]:
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
