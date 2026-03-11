"""
Canonical timestamp normalization for the market data pipeline.

Single source of truth for converting any timestamp representation to
timezone-aware UTC datetime. Replaces the scattered _normalize_ts()
implementations across market_data_service, angel_one_ws_connector,
normalizer, and yfinance_fallback_feeder.

All public functions return timezone-aware ``datetime`` in UTC.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Union

__all__ = ["normalize_ts", "IST", "UTC"]

# Named timezone constants
UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))


def normalize_ts(
    ts: Any,
    source_tz: str = "Asia/Kolkata",
) -> datetime:
    """Convert any timestamp representation to timezone-aware UTC datetime.

    Parameters
    ----------
    ts : None | datetime | str | int | float
        The raw timestamp value.
        - ``None``        -> current time in UTC.
        - naive datetime  -> assumed to be in *source_tz* and converted to UTC.
        - aware datetime  -> converted to UTC regardless of its original tz.
        - ``str``         -> parsed via ``datetime.fromisoformat`` then treated
                             as above (naive -> source_tz -> UTC).
        - ``int | float`` -> interpreted as Unix epoch seconds, returned as UTC.

    source_tz : str, default ``"Asia/Kolkata"``
        IANA timezone name used when the input is a naive (tz-unaware) datetime.
        Supported values are resolved to a fixed UTC offset internally to avoid
        a hard dependency on ``pytz`` / ``zoneinfo`` at import time.  The two
        most common values for Indian markets are:

        - ``"Asia/Kolkata"`` (IST, UTC+05:30)
        - ``"UTC"``

    Returns
    -------
    datetime
        Timezone-aware ``datetime`` in UTC.
    """
    if ts is None:
        return datetime.now(UTC)

    # Resolve source timezone to a fixed offset
    assumed_tz = _resolve_tz(source_tz)

    # --- int / float  (Unix epoch) ---
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=UTC)

    # --- str ---
    if isinstance(ts, str):
        ts = _parse_str(ts)

    # --- datetime ---
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            # Naive: attach the assumed source timezone, then convert to UTC
            return ts.replace(tzinfo=assumed_tz).astimezone(UTC)
        # Aware: convert to UTC
        return ts.astimezone(UTC)

    # Fallback: unrecognised type -> current UTC time (safe default)
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Small lookup of fixed-offset timezones so we avoid importing zoneinfo/pytz
# at module level.  Only timezones actually used in this codebase are listed.
_TZ_OFFSETS = {
    "Asia/Kolkata": IST,
    "IST": IST,
    "UTC": UTC,
    "US/Eastern": timezone(timedelta(hours=-5)),
    "America/New_York": timezone(timedelta(hours=-5)),
}


def _resolve_tz(name: str) -> timezone:
    """Resolve a timezone name to a ``datetime.timezone`` fixed offset.

    Falls back to ``zoneinfo`` (Python 3.9+) if the name is not in the
    hardcoded lookup, and finally defaults to IST if nothing else works.
    """
    tz = _TZ_OFFSETS.get(name)
    if tz is not None:
        return tz
    try:
        from zoneinfo import ZoneInfo
        zi = ZoneInfo(name)
        # ZoneInfo objects are accepted by datetime directly, but we need a
        # fixed-offset ``timezone`` for consistency.  Approximate via current
        # UTC offset (good enough for the "naive timestamp assumption" case).
        now = datetime.now(tz=zi)
        return timezone(now.utcoffset())  # type: ignore[arg-type]
    except Exception:
        return IST  # safe default for Indian markets


def _parse_str(s: str) -> Union[datetime, str]:
    """Best-effort parse of an ISO-8601-ish string to datetime.

    Returns the original string unchanged if parsing fails so the caller
    can fall through to the fallback branch.
    """
    # Normalise common variants
    cleaned = s.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        pass

    # Try common formats
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(cleaned, fmt)
        except (ValueError, TypeError):
            continue

    return s  # unparsable -> caller falls through to fallback
