"""
Data Retention Manager for SEBI Compliance.

SEBI mandates a minimum 5-year retention period for all trading-related
records. This module enforces retention policies, manages archival of
old data to cold storage, and prevents deletion of records within the
mandatory retention window.

Key features:
  - Automated 5-year minimum retention enforcement.
  - Archive old data (compress, move to cold storage path).
  - Prevent deletion of records within retention window.
  - Retention policy configuration per data type.
  - Cleanup of data beyond maximum retention period.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEBI_MIN_RETENTION_YEARS = 5
DEFAULT_MAX_RETENTION_YEARS = 8  # Keep up to 8 years before purging


class DataCategory(str, Enum):
    """Categories of data with retention policies."""

    AUDIT_EVENTS = "AUDIT_EVENTS"
    TRADE_RECORDS = "TRADE_RECORDS"
    ORDER_RECORDS = "ORDER_RECORDS"
    RISK_DECISIONS = "RISK_DECISIONS"
    MARKET_DATA = "MARKET_DATA"
    MODEL_ARTIFACTS = "MODEL_ARTIFACTS"
    SURVEILLANCE_ALERTS = "SURVEILLANCE_ALERTS"
    OTR_RECORDS = "OTR_RECORDS"
    ALGO_REGISTRATIONS = "ALGO_REGISTRATIONS"
    REGULATORY_REPORTS = "REGULATORY_REPORTS"


class RetentionAction(str, Enum):
    """Actions taken on data by the retention manager."""

    RETAINED = "RETAINED"  # Data within retention window, kept
    ARCHIVED = "ARCHIVED"  # Compressed and moved to cold storage
    DELETION_BLOCKED = "DELETION_BLOCKED"  # Deletion attempt blocked
    PURGED = "PURGED"  # Beyond max retention, removed


@dataclass
class RetentionPolicy:
    """
    Retention policy for a specific data category.

    Parameters
    ----------
    category : DataCategory
        The data category this policy applies to.
    min_retention_years : int
        Minimum years to retain (SEBI minimum 5).
    max_retention_years : int
        Maximum years to retain before purging.
    archive_after_years : int
        Years after which data is moved to cold storage.
    compress_archive : bool
        Whether to gzip-compress archived data.
    """

    category: DataCategory
    min_retention_years: int = SEBI_MIN_RETENTION_YEARS
    max_retention_years: int = DEFAULT_MAX_RETENTION_YEARS
    archive_after_years: int = 2
    compress_archive: bool = True

    def __post_init__(self) -> None:
        if self.min_retention_years < SEBI_MIN_RETENTION_YEARS:
            raise ValueError(
                f"SEBI requires minimum {SEBI_MIN_RETENTION_YEARS}-year retention. "
                f"Got {self.min_retention_years} for {self.category.value}."
            )
        if self.max_retention_years < self.min_retention_years:
            raise ValueError(
                f"max_retention_years ({self.max_retention_years}) must be >= "
                f"min_retention_years ({self.min_retention_years})."
            )
        if self.archive_after_years > self.min_retention_years:
            self.archive_after_years = self.min_retention_years


@dataclass
class RetentionRecord:
    """Record of a retention action taken on a data item."""

    record_id: str
    category: DataCategory
    action: RetentionAction
    timestamp: datetime
    data_date: datetime
    description: str
    file_path: str | None = None
    archive_path: str | None = None


class RetentionManager:
    """
    Thread-safe data retention manager for SEBI compliance.

    Manages data lifecycle: retention, archival, and purging according
    to configurable per-category policies.

    Parameters
    ----------
    cold_storage_path : str or Path
        Path to cold storage directory for archived data.
    policies : list[RetentionPolicy], optional
        Custom retention policies. Defaults are created for all categories.
    """

    def __init__(
        self,
        cold_storage_path: str = "data/archive",
        policies: list[RetentionPolicy] | None = None,
    ) -> None:
        self._cold_storage = Path(cold_storage_path)
        self._cold_storage.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Initialize policies
        self._policies: dict[DataCategory, RetentionPolicy] = {}
        if policies:
            for p in policies:
                self._policies[p.category] = p

        # Ensure all categories have a policy (with SEBI-compliant defaults)
        for cat in DataCategory:
            if cat not in self._policies:
                self._policies[cat] = RetentionPolicy(category=cat)

        # Action log
        self._action_log: list[RetentionRecord] = []

    # ------------------------------------------------------------------
    # Public API: retention checks
    # ------------------------------------------------------------------

    def is_within_retention_window(
        self,
        category: DataCategory,
        data_date: datetime,
    ) -> bool:
        """
        Check if a data record is within the mandatory retention window.

        Parameters
        ----------
        category : DataCategory
            Category of the data.
        data_date : datetime
            Timestamp of the data record.

        Returns
        -------
        bool
            True if the data must be retained (cannot be deleted).
        """
        policy = self._policies[category]
        now = datetime.now(UTC)
        retention_end = data_date + timedelta(days=policy.min_retention_years * 365)
        return now < retention_end

    def can_delete(
        self,
        category: DataCategory,
        data_date: datetime,
    ) -> bool:
        """
        Check if a data record can be safely deleted.

        Returns False if the record is within the SEBI retention window.
        """
        return not self.is_within_retention_window(category, data_date)

    def block_deletion_if_required(
        self,
        category: DataCategory,
        data_date: datetime,
        record_description: str = "",
    ) -> bool:
        """
        Attempt to delete a record. Returns True if deletion is allowed,
        False if blocked by retention policy.

        Logs the action either way.
        """
        import uuid

        if self.is_within_retention_window(category, data_date):
            now = datetime.now(UTC)
            policy = self._policies[category]
            retention_end = data_date + timedelta(days=policy.min_retention_years * 365)

            record = RetentionRecord(
                record_id=str(uuid.uuid4()),
                category=category,
                action=RetentionAction.DELETION_BLOCKED,
                timestamp=now,
                data_date=data_date,
                description=(
                    f"Deletion BLOCKED for {category.value}: {record_description}. "
                    f"SEBI retention requires keeping until {retention_end.isoformat()}."
                ),
            )

            with self._lock:
                self._action_log.append(record)

            logger.warning(
                "RETENTION BLOCK: Cannot delete %s data from %s. Retention period ends %s.",
                category.value,
                data_date.isoformat(),
                retention_end.isoformat(),
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Public API: archival
    # ------------------------------------------------------------------

    def should_archive(
        self,
        category: DataCategory,
        data_date: datetime,
    ) -> bool:
        """Check if data should be moved to cold storage."""
        policy = self._policies[category]
        now = datetime.now(UTC)
        archive_threshold = data_date + timedelta(days=policy.archive_after_years * 365)
        return now >= archive_threshold

    def archive_data(
        self,
        category: DataCategory,
        data_date: datetime,
        data: Any,
        source_path: str | None = None,
    ) -> str | None:
        """
        Archive data to cold storage.

        Parameters
        ----------
        category : DataCategory
            Category of the data being archived.
        data_date : datetime
            Original timestamp of the data.
        data : any
            Data to archive (must be JSON-serializable or bytes).
        source_path : str, optional
            Path of the original file (for file-based archival).

        Returns
        -------
        str or None
            Path to the archived file, or None if archival was skipped.
        """
        import uuid

        if not self.should_archive(category, data_date):
            return None

        policy = self._policies[category]
        now = datetime.now(UTC)

        # Construct archive path: cold_storage / category / year / month /
        archive_dir = self._cold_storage / category.value.lower() / str(data_date.year) / f"{data_date.month:02d}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        date_str = data_date.strftime("%Y%m%d_%H%M%S")
        archive_id = str(uuid.uuid4())[:8]

        if source_path and os.path.isfile(source_path):
            # Archive an existing file
            base_name = os.path.basename(source_path)
            archive_name = f"{date_str}_{archive_id}_{base_name}"
            if policy.compress_archive:
                archive_name += ".gz"
                archive_path = archive_dir / archive_name
                with open(source_path, "rb") as f_in:
                    with gzip.open(str(archive_path), "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                archive_path = archive_dir / archive_name
                shutil.copy2(source_path, str(archive_path))
        else:
            # Archive in-memory data as JSON
            archive_name = f"{date_str}_{archive_id}_{category.value.lower()}.json"
            if policy.compress_archive:
                archive_name += ".gz"
                archive_path = archive_dir / archive_name
                json_bytes = json.dumps(data, default=str).encode("utf-8")
                with gzip.open(str(archive_path), "wb") as f_out:
                    f_out.write(json_bytes)
            else:
                archive_path = archive_dir / archive_name
                with open(str(archive_path), "w") as f_out:
                    json.dump(data, f_out, default=str, indent=2)

        record = RetentionRecord(
            record_id=str(uuid.uuid4()),
            category=category,
            action=RetentionAction.ARCHIVED,
            timestamp=now,
            data_date=data_date,
            description=f"Data archived to cold storage: {archive_path}",
            file_path=source_path,
            archive_path=str(archive_path),
        )

        with self._lock:
            self._action_log.append(record)

        logger.info(
            "ARCHIVED %s data from %s to %s%s",
            category.value,
            data_date.isoformat(),
            archive_path,
            " (compressed)" if policy.compress_archive else "",
        )

        return str(archive_path)

    # ------------------------------------------------------------------
    # Public API: purge old data
    # ------------------------------------------------------------------

    def should_purge(
        self,
        category: DataCategory,
        data_date: datetime,
    ) -> bool:
        """Check if data has exceeded the maximum retention period and can be purged."""
        policy = self._policies[category]
        now = datetime.now(UTC)
        max_retention_end = data_date + timedelta(days=policy.max_retention_years * 365)
        return now >= max_retention_end

    def purge_if_expired(
        self,
        category: DataCategory,
        data_date: datetime,
        file_path: str | None = None,
        record_description: str = "",
    ) -> bool:
        """
        Purge data if it has exceeded the maximum retention period.

        Returns True if the data was purged, False if it is still within retention.
        """
        import uuid

        # Double-check: never purge within minimum retention
        if self.is_within_retention_window(category, data_date):
            logger.warning(
                "Attempted to purge %s data from %s still within SEBI retention window.",
                category.value,
                data_date.isoformat(),
            )
            return False

        if not self.should_purge(category, data_date):
            return False

        now = datetime.now(UTC)

        # Remove the file if specified
        if file_path and os.path.isfile(file_path):
            os.remove(file_path)
            logger.info("PURGED file: %s", file_path)

        record = RetentionRecord(
            record_id=str(uuid.uuid4()),
            category=category,
            action=RetentionAction.PURGED,
            timestamp=now,
            data_date=data_date,
            description=(
                f"Data PURGED (exceeded {self._policies[category].max_retention_years}-year "
                f"max retention): {record_description}"
            ),
            file_path=file_path,
        )

        with self._lock:
            self._action_log.append(record)

        return True

    # ------------------------------------------------------------------
    # Public API: cleanup cycle
    # ------------------------------------------------------------------

    def run_cleanup_cycle(
        self,
        data_directory: str,
        category: DataCategory,
    ) -> dict[str, int]:
        """
        Run a full cleanup cycle on a data directory:
        1. Archive data older than archive_after_years.
        2. Purge data older than max_retention_years.

        Parameters
        ----------
        data_directory : str
            Path to directory containing data files.
        category : DataCategory
            Category of data in the directory.

        Returns
        -------
        dict
            Counts: {'archived': N, 'purged': N, 'retained': N, 'errors': N}
        """
        counts = {"archived": 0, "purged": 0, "retained": 0, "errors": 0}
        dir_path = Path(data_directory)

        if not dir_path.is_dir():
            logger.warning("Data directory does not exist: %s", data_directory)
            return counts

        for file_path in dir_path.iterdir():
            if not file_path.is_file():
                continue

            try:
                # Infer data date from file modification time
                mtime = file_path.stat().st_mtime
                data_date = datetime.fromtimestamp(mtime, tz=UTC)

                if self.should_purge(category, data_date):
                    if self.purge_if_expired(
                        category,
                        data_date,
                        file_path=str(file_path),
                        record_description=file_path.name,
                    ):
                        counts["purged"] += 1
                    else:
                        counts["retained"] += 1
                elif self.should_archive(category, data_date):
                    archive_path = self.archive_data(
                        category,
                        data_date,
                        data=None,
                        source_path=str(file_path),
                    )
                    if archive_path:
                        counts["archived"] += 1
                    else:
                        counts["retained"] += 1
                else:
                    counts["retained"] += 1

            except Exception as e:
                logger.error("Error processing %s: %s", file_path, e)
                counts["errors"] += 1

        logger.info(
            "Cleanup cycle for %s in %s: %s",
            category.value,
            data_directory,
            counts,
        )
        return counts

    # ------------------------------------------------------------------
    # Public API: query and reporting
    # ------------------------------------------------------------------

    def get_policies(self) -> list[dict[str, Any]]:
        """Get all configured retention policies."""
        return [
            {
                "category": p.category.value,
                "min_retention_years": p.min_retention_years,
                "max_retention_years": p.max_retention_years,
                "archive_after_years": p.archive_after_years,
                "compress_archive": p.compress_archive,
                "sebi_compliant": p.min_retention_years >= SEBI_MIN_RETENTION_YEARS,
            }
            for p in self._policies.values()
        ]

    def get_action_log(
        self,
        category: DataCategory | None = None,
        action: RetentionAction | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent retention actions."""
        with self._lock:
            log = list(self._action_log)

        if category:
            log = [r for r in log if r.category == category]
        if action:
            log = [r for r in log if r.action == action]

        log = log[-limit:]
        log.reverse()

        return [
            {
                "record_id": r.record_id,
                "category": r.category.value,
                "action": r.action.value,
                "timestamp": r.timestamp.isoformat(),
                "data_date": r.data_date.isoformat(),
                "description": r.description,
                "file_path": r.file_path,
                "archive_path": r.archive_path,
            }
            for r in log
        ]

    def get_retention_summary(self) -> dict[str, Any]:
        """Get a summary of retention status across all categories."""
        with self._lock:
            log = list(self._action_log)

        summary = {
            "sebi_min_retention_years": SEBI_MIN_RETENTION_YEARS,
            "total_actions": len(log),
            "actions_by_type": {},
            "policies": self.get_policies(),
        }

        for action_type in RetentionAction:
            count = sum(1 for r in log if r.action == action_type)
            if count > 0:
                summary["actions_by_type"][action_type.value] = count

        return summary
